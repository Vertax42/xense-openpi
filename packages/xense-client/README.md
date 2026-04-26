# xense-client

`xense-client` 是 OpenPI 项目提供的**异步远程策略推理客户端框架**，用于在真实机器人控制环路中消化神经网络推理的时延抖动。本包本身不包含模型实现，而是通过 WebSocket 连接远端策略服务端（`src/openpi/serving/websocket_policy_server.py`），把**推理、动作缓冲、时延补偿、控制循环**四件事彻底解耦。

## 目录与模块

```
packages/xense-client/src/xense_client/
├── base_policy.py              # BasePolicy 抽象接口
├── websocket_client_policy.py  # 远程策略客户端（msgpack + WebSocket）
├── action_chunk_broker.py      # 同步版 chunk 切片器
├── rtc_action_chunk_broker.py  # 异步 RTC 版（核心）
├── action_queue.py             # 线程安全动作队列
├── latency_tracker.py          # 延迟统计
├── msgpack_numpy.py            # numpy 友好的序列化
├── image_tools.py              # 图像预处理
└── runtime/                    # 控制循环骨架
    ├── agent.py / environment.py / subscriber.py
    ├── runtime.py
    └── agents/policy_agent.py
```

## 分层架构

```
┌──────────────────────────────────────────────────┐
│ Runtime (runtime.py)       控制循环 / 频率节拍       │
├──────────────────────────────────────────────────┤
│ Agent (PolicyAgent)        把 Policy 适配成 Agent   │
├──────────────────────────────────────────────────┤
│ RTCActionChunkBroker       异步推理 + 动作缓冲队列   │
├──────────────────────────────────────────────────┤
│ WebsocketClientPolicy      网络层（msgpack + WS）   │
├──────────────────────────────────────────────────┤
│ ActionQueue / msgpack_numpy / latency_tracker    │
└──────────────────────────────────────────────────┘
```

## 核心模块

### 1. BasePolicy（`base_policy.py`）

统一接口，三个方法：

- `infer(obs) -> dict`：核心推理
- `reset()`：episode 开始时清状态
- `warmup(obs)`：触发 JAX JIT 编译（关键，JAX 首次编译约 400ms）

下游策略都实现此接口并可互相装饰，形成 **Decorator 链**：
`RTCActionChunkBroker(WebsocketClientPolicy(...))`。

### 2. WebsocketClientPolicy（`websocket_client_policy.py`）

- 建连阻塞等待服务端（循环 sleep 5s），拿到第一条消息作为 **server metadata**（模型 / 动作空间元信息）。
- `infer(obs, **kwargs)`：若有 RTC 相关 kwargs，塞进 `obs["__rtc_kwargs__"]`（避免破坏协议），然后 `packer.pack → ws.send → ws.recv → unpackb`。字符串响应视为服务端错误。
- 可选 `api_key`（Authorization header），禁用压缩以降低延迟。

服务端对应 `websocket_policy_server.py`：收到后 `obs.pop("__rtc_kwargs__")` 传给 `policy.infer(obs, **rtc_kwargs)`，并在响应中附加 `server_timing.infer_ms`。

### 3. msgpack_numpy（`msgpack_numpy.py`）

**为什么不用 pickle？** 安全（无任意代码执行）、跨语言、比 pickle 快约 4 倍。

把 `np.ndarray` 编码为 `{__ndarray__, data(bytes), dtype, shape}`，解码时利用 `np.ndarray(buffer=..., dtype=..., shape=...)` 做**零拷贝**重建。`np.generic` 标量类似处理。

### 4. ActionChunkBroker（`action_chunk_broker.py`）—— 同步版

模型每次返回 `(action_horizon, action_dim)` 的动作序列。该 broker 只做一件事：**缓存整段，每个 step 返回第 t 个动作**，用完再触发下一次 `infer`。简单、阻塞、推理时机器人停等。

### 5. RTCActionChunkBroker（`rtc_action_chunk_broker.py`）—— 真正的推理框架核心

解决"推理阻塞"与"时延抖动"的关键设计：

#### (a) 异步线程 + 动作队列

- 主线程：`infer(obs)` 只做两件事——更新 `latest_obs`（带锁）、从 `ActionQueue.get()` 取一个动作返回。
- 后台线程 `_get_actions_loop`：当队列 `qsize <= action_queue_size_to_get_new_actions`（默认 20）时发起下一次远端推理，结果通过 `ActionQueue.merge()` 合入队列。
- 机器人控制环路永远不等推理。

#### (b) 两阶段 warmup

在 `warmup()` 里连发两次 `infer`：

1. 第一次：`prev_chunk_left_over=None`，触发 JAX JIT，结果**不入队**。
2. 第二次：`prev_chunk_left_over=phase1_result[-fixed_length:]`，验证 shape 并把结果入队。

目的：首个真实控制步不会再被 JIT 编译卡住。

#### (c) RTC（Real-Time Chunking）与保守时延估计

模型接收三个 RTC 参数：

- `prev_chunk_left_over`：尚未执行的旧动作（做 prefix 对齐，防止跳变）
- `inference_delay`：**估计**这次推理期间机器人会消耗多少动作，模型会把输出前 `inference_delay` 步冻结为旧动作的延续
- `execution_horizon`：本次要交付的执行长度

**保守 delay 估计的精髓**：

```python
if self._recent_real_delays:
    base_delay = max(self._recent_real_delays)            # 最近窗口取最大
    estimated_delay_steps = base_delay + self._delay_margin   # + 安全裕度
else:
    estimated_delay_steps = self._default_delay
estimated_delay_steps = min(estimated_delay_steps, current_qsize)
```

为什么不用"上一次真实延迟"？网络 / 调度抖动导致 real_delay 约 50% 时间超过上次值，超过即意味着模型冻结区间不够长，真实消费点落入"模型输出但未对齐"的区域 → **抖动 / 跳变**。取**滚动窗口最大值 + margin** 保证 `real_delay ≤ estimated_delay`，冻结前缀始终对齐。

#### (d) 动作合并与截断

`ActionQueue.merge(new_original, new_processed, estimated_delay, action_index_before_inference)`：

- `real_delay = 当前消费 index - 推理开始时的 index`
- `truncate_idx = min(estimated_delay, len(new))`（实现里结合 real_delay 做保守截断）
- 丢弃新 chunk 的前 `truncate_idx` 步（模型冻结的前缀），从中段接上队列。
- 可选 `blend_steps`：在接缝处做 α 线性混合，进一步抑制跳变。

### 6. ActionQueue（`action_queue.py`）

双队列：

- `original_queue`：未经后处理的原始动作，**用于 RTC 的 prev_chunk_left_over**
- `queue`：机器人真正执行的动作（可能经过缩放、滤波等后处理）

关键方法 `get_left_over(fixed_length)`：

```python
left_over = self.original_queue[self.last_index:]
if len(left_over) >= fixed_length:
    return left_over[:fixed_length]        # 头截断：保留最近待执行的
else:
    pad = np.repeat(left_over[-1:], fixed_length - len(left_over), axis=0)
    return np.concatenate([left_over, pad])  # 尾部重复填充
```

- **头截断**而非尾截断：模型只在前 `inference_delay` 步用它做冻结，必须保证前段真实对齐。
- **固定长度**：避免 JAX 因 shape 变化反复重编译。

### 7. LatencyTracker（`latency_tracker.py`）

`deque(maxlen=100)` 存最近延迟，提供 `percentile(q)` / `p95()`。注意**它只用于统计报表**，真正驱动时延估计的是 `_recent_real_delays`（真实被消费的 step 数），而非网络往返时间——这是个重要区分。

### 8. image_tools（`image_tools.py`）

两件事：`convert_to_uint8`（float → uint8，传输量降 4 倍）、`resize_with_pad`（保持宽高比的 letterbox，模拟 `tf.image.resize_with_pad`）。客户端侧完成以减轻带宽与服务端预处理压力。

### 9. Runtime 层（`runtime/runtime.py`）

经典的 **Environment + Agent + Subscriber** 三件套：

```python
def _run_episode(self):
    env.reset(); agent.reset()
    for s in subs: s.on_episode_start()
    warmup_obs = env.get_observation()
    agent.warmup(warmup_obs)     # 关键：触发 RTC 两阶段 warmup
    while in_episode:
        self._step()             # obs → action → apply → subscribers
        sleep 到目标频率
```

`_step()` 按阶段打点，便于定位 obs / infer / apply / subscriber 各自耗时。`PolicyAgent` 只是把 `agent.get_action` 直接转发给 `policy.infer`。

## 端到端时序

```
主线程 (50Hz)                          后台推理线程              服务端
────────────────────────────────────────────────────────────────────
env.get_observation
agent.get_action
  └─ queue.get() → a_t            (qsize<=20 触发)
env.apply_action(a_t)             latest_obs 读锁拷贝
sleep                             get_left_over(fixed_length)
                                  prev_chunk + 估 delay
                                  ws.send(msgpack(obs+rtc_kwargs)) ─► model.infer
                                                                  ◄── actions
                                  real_delay = 新 idx - 旧 idx
                                  ActionQueue.merge(截断 estimated_delay)
                                  _recent_real_delays.append(real_delay)
```

主线程**从不等网络**；后台线程的作用是"在队列见底前把下一段动作补齐，并让模型知道我消耗到哪儿了、大概还要多久"。

## 架构亮点

| 设计 | 解决的问题 |
|---|---|
| 装饰器式策略链 | 网络层、分块、RTC 相互正交，可组合 |
| 异步推理 + 双阶段 warmup | 消除 JIT 首跳变与推理阻塞 |
| 保守 delay 估计（max + margin） | 抖动下保证冻结前缀对齐 |
| 双队列（original / processed） | RTC 需要原始动作做 prefix，机器人需后处理动作 |
| 固定长度 leftover（头截断 / 尾 pad） | 避免 JAX 重编译 + 保持对齐 |
| msgpack + numpy 零拷贝 | 安全、跨语言、比 pickle 快 4× |
| Runtime 三件套 | 环境 / 策略 / 观测者解耦，方便接不同机器人 |

## 最小使用示例

```python
from xense_client.websocket_client_policy import WebsocketClientPolicy
from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker
from xense_client.runtime.agents.policy_agent import PolicyAgent
from xense_client.runtime.runtime import Runtime

policy = WebsocketClientPolicy(host="0.0.0.0", port=9000)
policy = RTCActionChunkBroker(
    policy=policy,
    frequency_hz=50.0,
    action_queue_size_to_get_new_actions=20,
    default_delay=4,
    delay_margin=2,
    execution_horizon=20,
)
agent = PolicyAgent(policy=policy)
runtime = Runtime(environment=my_env, agent=agent, subscribers=[], max_hz=50)
runtime.run()
```
