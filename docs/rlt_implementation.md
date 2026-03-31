# RLT (Reinforcement Learning Token) 实现文档

## 1. 概述

本实现基于 Physical Intelligence 论文 "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"，在 openpi 代码库中添加了 RLT 模块。

**核心思想**：在冻结的 VLA（π₀）模型上添加一个轻量级 RL 接口——通过编码器将 VLA 的高维内部表征压缩为一个紧凑的 "RL Token"（z_rl），然后在这个低维表征上训练小型 actor-critic 网络进行在线强化学习。

**关键设计约束**：**不修改 openpi 任何现有文件**，VLA 模型完全冻结，所有新代码在独立的 `src/openpi/rlt/` 模块中。

---

## 2. 文件结构

```
src/openpi/rlt/                      # 新增模块（10个文件）
├── __init__.py                      # 模块入口，导出公开 API
├── config.py                        # 所有配置 dataclass
├── vla_interface.py                 # 冻结 VLA 包装器（与 openpi 的对接层）
├── encoder_decoder.py               # RL Token 编码器 g_φ + 解码器 d_φ
├── actor.py                         # 高斯 Actor π_θ
├── critic.py                        # Twin Q Critic Q_ψ
├── replay_buffer.py                 # 离策略回放缓冲区
├── td3.py                           # TD3 算法（论文主算法）
├── ppo_wrapper.py                   # RSL-RL PPO 适配器（备选算法）
scripts/
└── train_rlt.py                     # 训练入口脚本
pyproject.toml                       # 修改：添加可选依赖 rlt 组
```

---

## 3. 与 openpi 的对接点（精确到代码行）

### 3.1 VLA 模型加载

**我们的代码** `scripts/train_rlt.py:66-97` `build_vla_model()` 函数：

```
第 73 行：调用 openpi.training.config.get_config(config_name)
         → 获取 openpi 的 TrainConfig，从中提取 model 配置（Pi0Config）
第 76-85 行：如果 model_cfg 不是 Pi0Config，则构造一个 Pi0Config
         → 使用 openpi 的 Pi0Config dataclass（来自 src/openpi/models/pi0_config.py:19）
第 87 行：openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
         → 实例化 openpi 的 PyTorch π₀ 模型（来自 src/openpi/models_pytorch/pi0_pytorch.py:88）
第 90-92 行：safetensors.torch.load_model(pi0_model, model_path)
         → 加载权重，与 openpi 的 train_pytorch.py:446 使用相同的 safetensors 格式
第 97 行：return VLAEmbeddingExtractor(pi0_model)
         → 用我们的包装器冻结 VLA
```

### 3.2 VLA Embedding 提取 — 最关键的对接层

**文件** `src/openpi/rlt/vla_interface.py`

**`VLAEmbeddingExtractor.__init__`（第 23-29 行）**：
```python
self.pi0 = pi0_model          # 持有 openpi 的 PI0Pytorch 实例
self.pi0.eval()                # 冻结为推理模式
for param in self.pi0.parameters():
    param.requires_grad = False  # 禁止所有梯度
```

**`extract_embeddings()`（第 31-69 行）**—— 数据流图：

```
                          openpi 代码                           我们的代码
                          ─────────                           ─────────
观测数据 observation
    │
    ▼
第 43 行: self.pi0._preprocess_observation(observation, train=False)
    │     → 调用 openpi pi0_pytorch.py:165-174
    │     → 返回 (images, img_masks, lang_tokens, lang_masks, state)
    │       images: 3个 [B,224,224,3] 相机图像
    │       state: [B, 32] 本体感受状态
    ▼
第 46-48 行: self.pi0.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    │     → 调用 openpi pi0_pytorch.py:190-239
    │     → SigLIP 编码图像（每张 ~192 个 token，2048维）
    │     → 语言 token 嵌入（48 或 200 个 token，2048维）
    │     → 返回 prefix_embs [B, M, 2048], prefix_pad_masks, prefix_att_masks
    │       M ≈ 3×192 + 48 = 624（典型值）
    ▼
第 51-53 行: 构建注意力掩码
    │     → make_att_2d_masks()  来自 openpi pi0_pytorch.py:56-85
    │     → pi0._prepare_attention_masks_4d()  来自 openpi pi0_pytorch.py:160-163
    ▼
第 56-58 行: 获取模型精度并转换
    │     → 读取 paligemma 第一层 q_proj 权重的 dtype（通常 bfloat16）
    ▼
第 61-67 行: self.pi0.paligemma_with_expert.forward(
                inputs_embeds=[prefix_embs, None],  ← None 表示只跑 prefix 分支
                use_cache=False,
            )
    │     → 调用 openpi gemma_pytorch.py:91-281
    │     → 走 gemma_pytorch.py:102-113 的 prefix-only 分支：
    │       - 只运行 PaliGemma 语言模型（18层 Gemma 2B transformer）
    │       - 不运行 action expert（节省计算）
    │     → 返回 prefix_output [B, M, 2048]（最终层 embedding）
    ▼
第 69 行: return prefix_output.float(), prefix_pad_masks, state
         → 输出送给 RL Token 编码器
```

**为什么选择 prefix-only 路径**：
- 论文中 z_{1:M} 的维度是 2048，与 PaliGemma 2B 的 width 匹配
  （参见 `src/openpi/models/gemma.py:83` → `width=2048`）
- action expert 的维度是 1024（`gemma.py:72` → `width=1024`），不匹配
- prefix 包含了完整的观测表征（图像+语言），而 suffix 需要动作输入
- prefix-only 路径不需要运行扩散去噪，速度更快

**`get_reference_actions()`（第 71-83 行）**：
```
第 83 行: self.pi0.sample_actions(device, observation, num_steps=10)
         → 调用 openpi pi0_pytorch.py:388-432
         → 运行完整的 10 步扩散去噪：
           1. 初始化噪声 x_t [B, 50, 32]
           2. embed_prefix() 一次（缓存 KV cache）
           3. 循环 10 次 denoise_step()（每次跑 embed_suffix + action expert）
           4. 返回去噪后的动作 [B, 50, 32]（H=50 步的动作块）
         → RL actor 从中切片前 C=10 步: vla_actions[:, :10, :14]
```

### 3.3 数据管道复用

**文件** `scripts/train_rlt.py:119-123`：

```python
第 120 行: train_config = _train_config.get_config(config.vla_config_name)
          → 调用 openpi training/config.py 的 get_config()
          → 获取预定义的训练配置（如 "pi0_droid"、"pi05_base_arx5_lora" 等）
          → 配置定义在 training/config.py:603+ 的 _CONFIGS 列表中

第 122 行: train_config = dataclasses.replace(train_config, batch_size=config.phase1_batch_size)
          → 覆盖 batch_size 为 RLT 的配置值

第 123 行: loader = _data.create_data_loader(train_config, framework="pytorch", shuffle=True)
          → 调用 openpi training/data_loader.py 的 create_data_loader()
          → 完全复用 openpi 的数据管道：
            - LeRobot 格式数据集加载
            - 机器人平台特定的数据转换（repack transforms）
            - 归一化处理
            - 模型特定的数据转换（tokenization 等）
          → 返回迭代器，每次产出 (observation, actions) 元组
```

**Phase 1 训练循环中使用数据**（`train_rlt.py:142-160`）：
```python
第 143 行: for observation, _actions in loader:
          → observation 包含 .images, .state, .tokenized_prompt 等
          → _actions 是 [B, 50, 32] 的动作标签（Phase 1 不需要）

第 148 行: observation = jax.tree.map(lambda x: x.to(device), observation)
          → 使用 jax.tree.map 处理嵌套结构（openpi 的 Observation 是嵌套 dict）
          → 这是 openpi 训练脚本的标准做法（参见 train_pytorch.py:520）
```

### 3.4 配置系统对接

**文件** `src/openpi/rlt/config.py`

遵循 openpi 的 frozen dataclass 模式（与 `training/config.py:509` 的 `TrainConfig` 一致）：
```python
@dataclasses.dataclass(frozen=True)  # 与 openpi 保持一致
class RLTTrainConfig:
    ...
```

但 `RLTTrainConfig` 是独立的（不继承 `TrainConfig`），因为 RL 训练循环与监督学习根本不同。通过 `vla_config_name` 字段引用 openpi 的配置：

```python
第 95 行: vla_config_name: str = "pi0_droid"  # openpi 配置名，对应 _CONFIGS 列表中的条目
```

---

## 4. 各模块详解

### 4.1 RL Token 编码器 — `encoder_decoder.py:12-70`

**对应论文公式 (1)**：z_rl = g_φ([z_{1:M}, e_rl])_{M+1}

```
RLTokenEncoder
├── rl_token_embedding: nn.Parameter [1, 1, 2048]     # 可学习的特殊 token e_rl（第 26 行）
├── transformer: nn.TransformerEncoder                  # 2 层，8 头，pre-norm（第 29-38 行）
└── final_norm: nn.LayerNorm(2048)                      # 最终归一化（第 39 行）

forward() 数据流（第 41-70 行）：
  输入: vla_embeddings [B, M, 2048], pad_mask [B, M]
  第 54 行: 扩展 e_rl → [B, 1, 2048]
  第 55 行: tokens = cat([vla_embeddings, e_rl]) → [B, M+1, 2048]   # 拼接到序列末尾
  第 58-59 行: 扩展 pad_mask → [B, M+1]
  第 62 行: src_key_padding_mask = ~extended_mask                     # PyTorch 用 True=忽略
  第 65 行: out = transformer(tokens, mask)                           # 全序列自注意力
  第 66 行: out = final_norm(out)
  第 69 行: z_rl = out[:, -1, :]  → [B, 2048]                        # 取最后位置 = RL token
```

### 4.2 RL Token 解码器 — `encoder_decoder.py:73-170`

**对应论文公式 (2)**：L_rto = E_D[Σ ||h_φ(d_φ([z_rl, z̃_{1:i-1}]))_i - z̃_i||²]

```
RLTokenDecoder
├── pos_embedding: nn.Embedding(1024, 2048)            # 位置编码（第 87 行）
├── transformer: nn.TransformerDecoder                  # 2 层因果解码器（第 90-99 行）
├── final_norm: nn.LayerNorm(2048)                      # 最终归一化（第 100 行）
└── output_head: nn.Linear(2048, 2048)                  # h_φ 线性输出头（第 103 行）

forward() 数据流（第 105-147 行）：
  输入: z_rl [B, D], target_embeddings [B, M, D]（stop-gradient 的 VLA embedding）
  第 119-120 行: shifted_input = [z_rl, z̃_1, ..., z̃_{M-1}]   # 自回归 teacher forcing
  第 123-124 行: 加位置编码
  第 127 行: memory = z_rl [B, 1, D]                             # 解码器的 memory = RL token
  第 130-132 行: causal_mask = 下三角掩码                         # 保证自回归
  第 138-143 行: 通过 transformer decoder
  第 145 行: reconstructed = output_head(out) → [B, M, D]        # 重建 VLA embedding

compute_loss()（第 149-170 行）：
  第 160 行: target_detached = target_embeddings.detach()         # stop-gradient
  第 164 行: error = (reconstructed - target_detached)²
  第 166-168 行: 用 pad_mask 掩码后求平均 MSE
```

### 4.3 Actor — `actor.py:13-108`

**对应论文公式 (4)**：π_θ(a_{1:C} | x, ã_{1:C}) = N(μ_θ(x, ã_{1:C}), σ²I)

```
RLTActor
├── mlp: Sequential                                     # 2 层 MLP [256, 256]（第 34-46 行）
│   输入维度 = 2048 + 32 + 10×14 = 2220
│   输出维度 = 10×14 = 140
└── log_std: Parameter [140]（固定，不学习）              # 固定标准差 0.1（第 49-52 行）

forward()（第 54-91 行）：
  输入: z_rl [B, 2048], proprio [B, 32], ref_actions [B, 10, 14]

  第 74 行: ref_flat = ref_actions.reshape(bsz, -1)      # [B, 140]
  第 77-79 行: Reference Action Dropout —— 论文的关键设计：
    以 50% 概率将整个 ref_flat 置零（逐样本，不是逐元素）
    → 迫使 actor 维持独立的动作生成能力，不只是复制 VLA 动作
  第 81 行: x = cat([z_rl, proprio, ref_flat])            # [B, 2220]
  第 82 行: mu = mlp(x)                                   # [B, 140]
  第 86 行: dist = Normal(mu, exp(log_std))               # 固定方差高斯
  第 87 行: actions_flat = dist.rsample()                  # 重参数化采样
  第 88 行: log_probs = dist.log_prob(...).sum(dim=-1)    # [B]
  第 90 行: actions = reshape → [B, 10, 14]               # 恢复 chunk 形状
```

### 4.4 Critic — `critic.py:12-69`

**对应论文公式 (3) 中的 Q_ψ(x, a_{1:C})**

```
QNetwork（单个 Q 网络，第 12-47 行）：
  输入维度 = 2048 + 32 + 10×14 = 2220
  MLP [256, 256] → 1                                     # 输出标量 Q 值

RLTCritic（Twin Q，第 50-69 行）：
  self.q1 = QNetwork(config)
  self.q2 = QNetwork(config)
  → 两个独立的 Q 网络，用于 TD3 的 clipped double-Q learning

  q1_forward()（第 67-69 行）：
    只计算 Q1（actor loss 中使用，省一半计算量）
```

### 4.5 Replay Buffer — `replay_buffer.py:12-102`

```
ReplayBuffer（第 12-102 行）：
  预分配 CPU 张量（第 23-30 行）：
    z_rl:        [capacity, 2048]     # RL token
    proprio:     [capacity, 32]       # 本体感受状态
    actions:     [capacity, 10, 14]   # 执行的动作块
    ref_actions: [capacity, 10, 14]   # VLA 参考动作块
    rewards:     [capacity, 1]        # chunk 级奖励（C 步累积）
    next_z_rl:   [capacity, 2048]     # 下一状态 RL token
    next_proprio:[capacity, 32]       # 下一状态本体感受
    dones:       [capacity, 1]        # episode 终止标志

  每个 transition 代表一个 action chunk 的执行（C=10 个时间步）

  add()（第 35-77 行）：支持单条和批量添加，环形缓冲
  sample()（第 79-99 行）：随机采样并移动到 GPU
```

### 4.6 TD3 算法 — `td3.py:21-189`

**对应论文 Algorithm 1 的第 13-18 行**

```
TD3.__init__（第 24-52 行）：
  第 41-42 行: 深拷贝 actor/critic 为 target 网络
  第 47-48 行: 分别创建 actor/critic 的 Adam 优化器
  第 52 行: _chunk_discount = γ^C = 0.99^10 ≈ 0.904
            → chunk 级折扣（不是单步 γ=0.99）

TD3.update()（第 61-134 行）—— 核心训练步：

  Critic 更新（每次都执行，第 83-109 行）：
    第 86 行: next_actions = actor_target(next_z_rl, next_proprio, ref_actions)
             → target 策略平滑
    第 88-90 行: 加截断噪声 ε ~ clip(N(0, 0.2), -0.5, 0.5)
    第 93-94 行: target_q = min(Q1_target, Q2_target)      # clipped double Q
    第 97 行: td_target = r + (1-done) × γ^C × target_q     # chunk 级 TD 目标
    第 100-101 行: critic_loss = MSE(Q1, target) + MSE(Q2, target)

  Actor 更新（每 2 次 critic 更新执行 1 次，第 111-132 行）：
    第 113 行: if _update_count % policy_delay == 0:        # 延迟策略更新
    第 115 行: pred_actions = actor(z_rl, proprio, ref_actions, apply_ref_dropout=True)
    第 116 行: q_value = critic.q1_forward(...)              # 只用 Q1
    第 119 行: bc_loss = MSE(pred_actions, ref_actions)
    第 120 行: actor_loss = -Q.mean() + β × bc_loss
             → 论文公式 (5)：L_π = E[-Q_ψ(x,a) + β||a - ã||²]
             → β 控制 actor 与 VLA 参考动作的接近程度
    第 131-132 行: 软更新 target 网络（τ=0.005）
```

### 4.7 PPO 适配器 — `ppo_wrapper.py`

**为 RSL-RL 库提供接口**

```
RLTValueNetwork（第 12-35 行）：
  V(z_rl, proprio) → 标量状态价值（PPO 需要，TD3 不需要）

RLTPPOActorCritic（第 38-136 行）：
  关键挑战：RSL-RL 期望扁平化的观测向量，但我们有三部分输入

  pack_observation()（第 68-82 行）：
    将 (z_rl, proprio, ref_actions) 打包为一个扁平向量
    → [B, 2048+32+140] = [B, 2220]

  _unpack()（第 55-66 行）：
    反向拆分：根据已知维度偏移切分扁平向量
    → z_rl = obs[:, :2048]
    → proprio = obs[:, 2048:2080]
    → ref_actions = obs[:, 2080:].reshape(-1, 10, 14)

  act()（第 84-106 行）：RSL-RL rollout 接口
  evaluate()（第 108-136 行）：RSL-RL PPO 更新接口

create_ppo_algorithm()（第 139-173 行）：
  第 153 行: from rsl_rl.algorithms import PPO  # 延迟导入（可选依赖）
  第 154-160 行: 若未安装则抛出 ImportError + 安装指令
```

---

## 5. 训练流程

### 5.1 Phase 1：RL Token 训练

**文件** `scripts/train_rlt.py:100-207` `train_phase1()`

```
步骤：
1. 加载冻结 VLA（第 113 行）→ build_vla_model()
2. 创建编码器-解码器（第 116 行）→ RLTokenEncoderDecoder(config.rl_token)
3. 复用 openpi 数据管道（第 120-123 行）→ create_data_loader()
4. 训练循环（第 142-187 行）：
   对每个 batch：
     a. 从 VLA 提取 embedding（第 151 行）→ vla.extract_embeddings(observation)
     b. 编码器压缩 + 解码器重建（第 154 行）→ enc_dec(vla_embeddings, pad_mask)
     c. 反传重建损失 L_rto（第 157-160 行）
5. 保存 encoder 权重（第 192-201 行）
```

### 5.2 Phase 2：在线 RL 训练

**文件** `scripts/train_rlt.py:248-335` `demo_phase2_td3()`

```
步骤：
1. 加载 Phase 1 的 encoder（第 260-272 行）→ 冻结
2. 创建 actor + critic + replay buffer + TD3（第 274-275 行）
3. 环境交互循环（实际使用时需用户提供环境）：
   在每个 action chunk 边界：
     a. VLA 提取 z_rl + 参考动作 ã
     b. actor 选择动作 a = π_θ(·|z_rl, proprio, ã)
     c. 环境执行 C=10 步，收集奖励
     d. 存储 transition 到 replay buffer
     e. 执行 UTD=5 次梯度更新（第 311-314 行）
```

---

## 6. 数据流总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                    openpi 代码（完全不修改）                          │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │ SigLIP      │    │ PaliGemma 2B │    │ Gemma 300M           │    │
│  │ 视觉编码器  │───▶│ 语言模型     │    │ 动作专家             │    │
│  │ pi0:200-206 │    │ gemma:103    │    │ (推理时使用)          │    │
│  └─────────────┘    └──────┬───────┘    └──────────────────────┘    │
│                            │                                         │
│                    prefix_output                                     │
│                    [B, M, 2048]                                      │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
              ┌──────────────┼────────── vla_interface.py:61-67
              │              ▼
              │   ┌────────────────────┐
              │   │ RLTokenEncoder     │   encoder_decoder.py:12-70
              │   │ 2层 Transformer    │
              │   │ [B,M,2048]→[B,2048]│
              │   └────────┬───────────┘
              │            │ z_rl [B, 2048]
              │            │
              │   ┌────────┴───────────────────────────────────┐
              │   │                                             │
              │   ▼                                             ▼
              │ ┌──────────────────┐              ┌──────────────────┐
              │ │ RLTActor         │              │ RLTCritic        │
              │ │ MLP [256,256]    │              │ Twin Q MLP       │
              │ │ actor.py:13-108  │              │ critic.py:12-69  │
              │ │                  │              │                  │
              │ │ 输入:            │              │ 输入:            │
              │ │  z_rl [2048]     │              │  z_rl [2048]     │
              │ │  proprio [32]    │              │  proprio [32]    │
              │ │  ref_act [10×14] │              │  actions [10×14] │
              │ │                  │              │                  │
              │ │ 输出:            │              │ 输出:            │
              │ │  actions [10,14] │              │  Q1, Q2 [1]      │
              │ └──────────────────┘              └──────────────────┘
              │
              │        openpi sample_actions()
              │        pi0_pytorch.py:388-432
              │              │
              │              ▼
              │   reference actions ã
              │   [B, 50, 32] → 切片 [:, :10, :14]
              │              │
              │              ▼
              │   传给 actor 作为条件输入
              └────────────────────────────────────────────────────────
```

---

## 7. 论文公式与代码的对应关系

| 论文公式 | 代码位置 | 说明 |
|---------|---------|------|
| 公式(1): z_rl = g_φ([z_{1:M}, e_rl])_{M+1} | `encoder_decoder.py:41-70` | 编码器前向传播 |
| 公式(2): L_rto = E[Σ\|\|h_φ(d_φ(...))_i - z̃_i\|\|²] | `encoder_decoder.py:149-170` | 解码器重建损失 |
| 公式(3): L_Q = E[(Q̂ - Q_ψ)²] | `td3.py:83-105` | TD3 critic 损失 |
| 公式(4): π_θ = N(μ_θ(x, ã), σ²I) | `actor.py:54-91` | 高斯 actor |
| 公式(5): L_π = E[-Q + β\|\|a-ã\|\|²] | `td3.py:114-120` | actor 损失 + BC 正则化 |
| Algorithm 1 第 7 行: 参考动作采样 | `vla_interface.py:71-83` | 调用 VLA sample_actions() |
| Algorithm 1 第 8 行: x_t = (z_rl, s^p) | `td3.py:74-75` | 从 replay buffer 取出 |
| Algorithm 1 第 15 行: TD backup Q̂ | `td3.py:93-97` | chunk 级折扣 γ^C |
| Reference action dropout | `actor.py:76-79` | 50% 概率置零 ref_actions |

---

## 8. 关键超参数及其含义

| 参数 | 默认值 | 来源 | 含义 |
|------|--------|------|------|
| `vla_embed_dim` | 2048 | `gemma.py:83` gemma_2b.width | VLA embedding 维度 |
| `action_dim` | 14 | 论文 Appendix B | 每时间步动作维度（7自由度×2臂） |
| `action_chunk` C | 10 | 论文 Section IV.B | RL 动作块长度（vs VLA 的 H=50） |
| `action_horizon` H | 50 | `pi0_config.py:28` | VLA 生成的完整动作块长度 |
| `proprio_dim` | 32 | `pi0_config.py:27` action_dim | 本体感受维度（openpi 内部用 32，含 padding） |
| `reference_dropout` | 0.5 | 论文 Section IV.B | 参考动作遮蔽概率 |
| `beta` | 1.0 | 论文公式(5) | BC 正则化系数 |
| `discount` γ | 0.99 | 论文标准设置 | 折扣因子（实际用 γ^C = 0.99^10 ≈ 0.904） |
| `utd_ratio` | 5 | 论文 Section V | 每步环境交互做 5 次梯度更新 |
| `policy_delay` | 2 | TD3 标准设置 | 每 2 次 critic 更新才更新 1 次 actor |

---

## 9. 使用方式

```bash
# Phase 1: 训练 RL Token 编码器-解码器
mamba run -n lerobot-xense python scripts/train_rlt.py \
    --phase 1 \
    --config.vla-checkpoint-path /path/to/pi0_checkpoint \
    --config.vla-config-name pi0_droid

# Phase 2: TD3 训练（需要用户提供环境交互）
mamba run -n lerobot-xense python scripts/train_rlt.py \
    --phase 2 \
    --config.rl-algorithm td3

# Phase 2: PPO 训练（需要安装 rsl-rl-lib）
pip install rsl-rl-lib>=5.0.1
mamba run -n lerobot-xense python scripts/train_rlt.py \
    --phase 2 \
    --config.rl-algorithm ppo
```

---

## 10. 测试验证

烟雾测试结果（`mamba run -n lerobot-xense`）：

```
Actor params:          671,512      # 轻量级，~0.7M
Critic params:       1,271,298      # Twin Q，~1.3M
EncoderDecoder params: 174,190,592  # 主要是 2048 维 transformer，~174M

Actor output:  [4, 10, 14]   →  B=4, C=10步动作块, d=14维动作
Critic output: [4, 1]        →  B=4, 标量 Q 值
Encoder output:[4, 2048]     →  B=4, D=2048 维 RL token
TD3 update:    critic_loss=1.94, q1_mean=0.69  →  正常训练
```
