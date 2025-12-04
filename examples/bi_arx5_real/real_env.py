import collections
import logging
import sys
import time
from typing import List, Optional

import dm_env
import numpy as np

# 添加 lerobot-ARX5 路径
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

# 导入 lerobot bi_arx5 (直接基于 ARX5 SDK，无ROS)
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config
from lerobot.robots.utils import make_robot_from_config

logger = logging.getLogger(__name__)

# 默认重置位置 (关节角度 + 夹爪位置) - 与你的 lerobot 配置保持一致
DEFAULT_RESET_POSITION = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.002]

# 安全限位 (底层限位的 0.9 倍作为检测阈值)
_JOINT_POS_MIN_RAW = np.array([-3.14, -0.05, -0.2, -1.6, -1.57, -2.0])
_JOINT_POS_MAX_RAW = np.array([2.618, 3.5, 3.2, 1.55, 1.57, 2.0])
_SAFETY_FACTOR = 0.9

# 计算安全限位 (使用 0.9 倍的范围)
JOINT_POS_MIN = _JOINT_POS_MIN_RAW * _SAFETY_FACTOR
JOINT_POS_MAX = _JOINT_POS_MAX_RAW * _SAFETY_FACTOR
GRIPPER_MIN = -0.1
GRIPPER_MAX = 1.8


class BiARX5RealEnv:
    """
    Environment for real BiARX5 robot bi-manual manipulation
    Based on lerobot BiARX5 implementation (No ROS, direct ARX5 SDK)

    Action space:      [left_joint_1.pos, ..., left_joint_6.pos, left_gripper.pos,
                        right_joint_1.pos, ..., right_joint_6.pos, right_gripper.pos]  # 14 dims

    Observation space: {"qpos": Concat[left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)],
                        "qvel": similar structure (zeros for BiARX5),
                        "images": {"cam_high": (H,W,3), "cam_left_wrist": (H,W,3), "cam_right_wrist": (H,W,3)}}
    """

    def __init__(
        self,
        left_arm_port: str = "can1",
        right_arm_port: str = "can3",
        log_level: str = "INFO",
        use_multithreading: bool = True,
        enable_tactile_sensors: bool = False,
        reset_position: Optional[List[float]] = None,
        setup_robot: bool = True,
        controller_dt: float = 0.002,  # 底层控制频率 (秒)
        preview_time: float = 0.02,  # 预览时间 (秒)
    ):
        self._reset_position = (
            reset_position if reset_position else DEFAULT_RESET_POSITION
        )

        # 创建 BiARX5 配置 - 直接使用你现有的 lerobot 配置
        self.config = BiARX5Config(
            id="bi_arx5_real_inference",
            left_arm_model="X5",
            left_arm_port=left_arm_port,
            right_arm_model="X5",
            right_arm_port=right_arm_port,
            log_level=log_level,
            use_multithreading=use_multithreading,
            enable_tactile_sensors=enable_tactile_sensors,
            inference_mode=True,  # 推理模式，设置合适的 preview_time
            controller_dt=controller_dt,  # 从参数传入
            preview_time=preview_time,  # 从参数传入
        )

        # 创建机器人实例 - 使用你现有的 BiARX5 类
        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self):
        """连接并初始化机器人 (无ROS，直接ARX5 SDK)"""
        logger.info("Connecting to BiARX5 robot via ARX5 SDK...")
        try:
            self.robot.connect(calibrate=False, go_to_start=True)
            logger.info("BiARX5 robot connected and ready for inference")
        except Exception as e:
            logger.error(f"Failed to connect BiARX5 robot: {e}")
            raise

    def get_qpos(self, obs):
        """获取关节位置 - 直接从你的 BiARX5.get_observation() 获取"""
        # 按照 aloha_real 的格式组织: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        left_joints = [obs[f"left_joint_{i+1}.pos"] for i in range(6)]
        left_gripper = [obs["left_gripper.pos"]]

        right_joints = [obs[f"right_joint_{i+1}.pos"] for i in range(6)]
        right_gripper = [obs["right_gripper.pos"]]

        return np.concatenate([left_joints, left_gripper, right_joints, right_gripper])

    def get_qvel(self, obs):
        """获取关节速度 - BiARX5 SDK 暂不提供速度反馈，返回零向量"""
        return np.zeros(14)  # 6+1 + 6+1 = 14

    def get_images(self, obs):
        """获取相机图像 - 直接从你的 BiARX5.get_observation() 获取"""
        images = {}

        # 映射你的相机名称到 aloha_real 期望的名称
        camera_mapping = {
            "head": "cam_high",  # 头部相机 -> cam_high
            "left_wrist": "cam_left_wrist",  # 左腕相机 -> cam_left_wrist
            "right_wrist": "cam_right_wrist",  # 右腕相机 -> cam_right_wrist
        }
        if self.config.enable_tactile_sensors:
            camera_mapping["left_tactile_0"] = "left_tactile_0"
            camera_mapping["right_tactile_0"] = "right_tactile_0"

        for lerobot_name, openpi_name in camera_mapping.items():
            if lerobot_name in obs:
                images[openpi_name] = obs[lerobot_name]
            else:
                logger.warning(f"Camera {lerobot_name} not found in observation")

        return images

    def get_observation(self):
        """获取完整观测 - 兼容 aloha_real 格式"""
        current_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(current_obs)
        obs["qvel"] = self.get_qvel(current_obs)
        obs["images"] = self.get_images(current_obs)
        # print(obs["images"].keys()) # display the keys of the robot observation images
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        """重置机器人到初始位置 - 使用你的 smooth_go_start()"""
        if not fake:
            logger.info("Resetting BiARX5 to start position...")
            try:
                # 使用你现有的平滑移动方法
                self.robot.smooth_go_start(duration=2.0)
                logger.info("BiARX5 reset completed")
            except Exception as e:
                logger.error(f"Failed to reset BiARX5: {e}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def _check_action_safety(self, action: np.ndarray) -> tuple[bool, str]:
        """检查动作是否在安全范围内
        
        Args:
            action: 14维动作数组 [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]
            
        Returns:
            (is_safe, error_message): 是否安全，以及错误信息
        """
        action = np.asarray(action)
        
        # 检查左臂关节 (index 0-5)
        left_joints = action[0:6]
        for i, (val, min_val, max_val) in enumerate(zip(left_joints, JOINT_POS_MIN, JOINT_POS_MAX)):
            if val < min_val or val > max_val:
                return False, f"Left joint {i+1} out of range: {val:.4f} not in [{min_val:.4f}, {max_val:.4f}]"
        
        # 检查左夹爪 (index 6)
        left_gripper = action[6]
        if left_gripper < GRIPPER_MIN or left_gripper > GRIPPER_MAX:
            return False, f"Left gripper out of range: {left_gripper:.4f} not in [{GRIPPER_MIN}, {GRIPPER_MAX}]"
        
        # 检查右臂关节 (index 7-12)
        right_joints = action[7:13]
        for i, (val, min_val, max_val) in enumerate(zip(right_joints, JOINT_POS_MIN, JOINT_POS_MAX)):
            if val < min_val or val > max_val:
                return False, f"Right joint {i+1} out of range: {val:.4f} not in [{min_val:.4f}, {max_val:.4f}]"
        
        # 检查右夹爪 (index 13)
        right_gripper = action[13]
        if right_gripper < GRIPPER_MIN or right_gripper > GRIPPER_MAX:
            return False, f"Right gripper out of range: {right_gripper:.4f} not in [{GRIPPER_MIN}, {GRIPPER_MAX}]"
        
        return True, ""

    def step(self, action):
        """执行动作 - 转换格式后调用你的 send_action()"""
        # 安全检查
        is_safe, error_msg = self._check_action_safety(action)
        if not is_safe:
            logger.error(f"Action safety check failed: {error_msg}")
            raise ValueError(f"Unsafe action detected: {error_msg}")
        
        # 确保机器人处于正常位置控制模式（而不是重力补偿模式）
        if self.robot.is_gravity_compensation_mode():
            logger.info(
                "Switching from gravity compensation to normal position control for action execution"
            )
            self.robot.set_to_normal_position_control()
        # else:
        #     logger.info("Robot is already in normal position control mode")

        # 将 OpenPI 的动作数组转换为你的 BiARX5 动作字典格式
        action_dict = {}

        # 左臂动作 (前7个: 6关节 + 夹爪)
        for i in range(6):
            action_dict[f"left_joint_{i+1}.pos"] = float(action[i])
        action_dict["left_gripper.pos"] = float(action[6])

        # 右臂动作 (后7个: 6关节 + 夹爪)
        for i in range(6):
            action_dict[f"right_joint_{i+1}.pos"] = float(action[7 + i])
        action_dict["right_gripper.pos"] = float(action[13])
        # for better gripper control to catch the cubes
        if action_dict["left_gripper.pos"] < 0.48 and action_dict["left_gripper.pos"] > 0.38:
            action_dict["left_gripper.pos"] = 0.4
        if action_dict["right_gripper.pos"] < 0.45 and action_dict["right_gripper.pos"] > 0.35:
            action_dict["right_gripper.pos"] = 0.4
        
        # print(
        #     "gripper_action",
        #     action_dict["left_gripper.pos"],
        #     action_dict["right_gripper.pos"],
        # )

        # 发送动作到你的 BiARX5 机器人
        try:
            self.robot.send_action(action_dict)
        except Exception as e:
            logger.error(f"Failed to send action to BiARX5: {e}")
            raise

        # 等待一个控制周期 (你的配置中 controller_dt = 0.01，即100Hz)
        # time.sleep(self.config.controller_dt)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def disconnect(self):
        """断开机器人连接"""
        if self.robot.is_connected:
            logger.info("Disconnecting BiARX5 robot...")
            try:
                self.robot.disconnect()
                time.sleep(1)
                logger.info("BiARX5 robot disconnected")
            except Exception as e:
                logger.warning(f"Error during BiARX5 disconnect: {e}")


def make_bi_arx5_real_env(
    left_arm_port: str = "can1",
    right_arm_port: str = "can3",
    log_level: str = "INFO",
    use_multithreading: bool = True,
    reset_position: Optional[List[float]] = None,
    enable_tactile_sensors: bool = False,
    setup_robot: bool = True,
    controller_dt: float = 0.002,
    preview_time: float = 0.02,
) -> BiARX5RealEnv:
    """创建 BiARX5 真实环境 (基于 lerobot 实现，无ROS版本)"""
    return BiARX5RealEnv(
        left_arm_port=left_arm_port,
        right_arm_port=right_arm_port,
        log_level=log_level,
        use_multithreading=use_multithreading,
        reset_position=reset_position,
        enable_tactile_sensors=enable_tactile_sensors,
        setup_robot=setup_robot,
        controller_dt=controller_dt,
        preview_time=preview_time,
    )
