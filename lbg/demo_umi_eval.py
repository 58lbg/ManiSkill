"""
UMI Policy Evaluation in ManiSkill Simulation

Usage:
python mani_skill/examples/demo_umi_eval.py \
    --robot_config=mani_skill/examples/config/eval_robots_config.yaml \
    -i cup_wild_vit_l.ckpt \
    -o data/eval_cup_wild_example

# 运行命令
python lbg/demo_umi_eval.py \
    --robot_config=lbg/config/eval_robots_config.yaml \
    -i lbg/train/'epoch=0000-train_loss=0.044.ckpt' \
    -o lbg/data/eval_cup_wild_example

Controls:
- Press 'C' to start policy evaluation
- Press 'S' to stop policy and regain manual control  
- Press 'Q' to quit
"""

import os
import sys
import time
from pathlib import Path
import click
import yaml
import dill
import hydra
import numpy as np
import torch
import cv2
import scipy.spatial.transform as st
from omegaconf import OmegaConf

import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common

# Import UMI utilities
sys.path.append(str(Path(__file__).parent.parent.parent / "universal_manipulation_interface-main"))
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from lbg.utils.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
from lbg.utils.pose_util import pose_to_mat, mat_to_pose

OmegaConf.register_new_resolver("eval", eval, replace=True)


class ObservationBuffer:
    """维护观测历史的缓冲区，用于提供真实的时序数据"""
    def __init__(self, history_len=2):
        self.history_len = history_len
        self.camera_history = []
        self.position_history = []
        self.rotation_history = []
        self.gripper_history = []
        self.timestamp_history = []
    
    def add(self, camera, position, rotation, gripper, timestamp):
        """添加新的观测"""
        self.camera_history.append(camera)
        self.position_history.append(position)
        self.rotation_history.append(rotation)
        self.gripper_history.append(gripper)
        self.timestamp_history.append(timestamp)
        
        # 保持固定长度
        if len(self.camera_history) > self.history_len:
            self.camera_history.pop(0)
            self.position_history.pop(0)
            self.rotation_history.pop(0)
            self.gripper_history.pop(0)
            self.timestamp_history.pop(0)
    
    def get_obs_dict(self):
        """获取格式化的观测字典"""
        # 如果历史不够，用最早的帧填充
        while len(self.camera_history) < self.history_len:
            if len(self.camera_history) > 0:
                self.camera_history.insert(0, self.camera_history[0])
                self.position_history.insert(0, self.position_history[0])
                self.rotation_history.insert(0, self.rotation_history[0])
                self.gripper_history.insert(0, self.gripper_history[0])
                self.timestamp_history.insert(0, self.timestamp_history[0] - 1/60)
            else:
                # 完全没有数据，返回None
                return None
        
        return {
            'camera0_rgb': np.stack(self.camera_history, axis=0),  # (T, 224, 224, 3)
            'timestamp': np.array(self.timestamp_history),  # (T,)
            'robot0_eef_pos': np.stack(self.position_history, axis=0),  # (T, 3)
            'robot0_eef_rot_axis_angle': np.stack(self.rotation_history, axis=0),  # (T, 3)
            'robot0_gripper_width': np.stack(self.gripper_history, axis=0),  # (T, 1)
        }
    
    def reset(self):
        """重置历史"""
        self.camera_history.clear()
        self.position_history.clear()
        self.rotation_history.clear()
        self.gripper_history.clear()
        self.timestamp_history.clear()


def get_sim_obs_dict(env, obs, obs_buffer):
    """
    从仿真环境获取当前观测并更新历史缓冲区
    
    Args:
        env: ManiSkill environment
        obs: current observation
        obs_buffer: ObservationBuffer instance
    
    Returns:
        obs_data: 包含真实历史观测的字典（非重复帧），如果数据不足返回None
    
    # 期待数据格式: 
    obs_data = {
        'camera0_rgb': np.stack([camera_images[-2], camera_images[-1]], axis=0),  # (2, 224, 224, 3)
        'robot0_eef_pos': np.stack([ee_position, ee_position], axis=0),  # (2, 3)
        'robot0_eef_rot_axis_angle': np.stack([ee_rot_axis_angle, ee_rot_axis_angle], axis=0),  # (2, 3)
        'robot0_gripper_width': np.stack([gripper_qpos, gripper_qpos], axis=0),  # (2, 1)
        'timestamp': np.array([current_time - 1/60, current_time]),  # (2,)
    }
    """
    current_time = time.time()
    
    # 获取相机图像
    camera_img = None
    if "sensor_data" in obs and "hand_camera" in obs["sensor_data"]:
        cam_data = obs["sensor_data"]["hand_camera"]
        if "Color" in cam_data:
            # Get RGBA image and convert to RGB
            color_img = common.to_numpy(cam_data["Color"][0])  # (H, W, 4)
            rgb_img = color_img[:, :, :3]  # (H, W, 3)
            
            # Resize to 224x224 for model input
            rgb_img_resized = cv2.resize(rgb_img, (224, 224))
            
            # Convert to float32 [0, 1] if needed
            if rgb_img_resized.dtype == np.uint8:
                rgb_img_resized = rgb_img_resized.astype(np.float32) / 255.0
            
            camera_img = rgb_img_resized
    
    if camera_img is None:
        camera_img = np.zeros((224, 224, 3), dtype=np.float32)
    
    # Get end-effector pose
    ee_pose = env.agent.tcp.pose
    ee_position = common.to_numpy(ee_pose.p[0])  # (3,)
    ee_quaternion = common.to_numpy(ee_pose.q[0])  # (4,) [w, x, y, z]
    
    # Convert quaternion to rotation axis-angle
    rot = st.Rotation.from_quat([ee_quaternion[1], ee_quaternion[2], ee_quaternion[3], ee_quaternion[0]])
    ee_rot_axis_angle = rot.as_rotvec()  # (3,)
    
    # Get gripper state (use sum or mean of two joints as gripper width)
    qpos = env.agent.robot.get_qpos()[0]  # (9,)
    gripper_qpos = qpos[-2:]  # (2,)
    gripper_width = np.array([gripper_qpos.sum()])  # (1,) - sum of two finger joints
    
    # 添加到历史缓冲区（这是真实的历史数据，不是重复的当前帧）
    obs_buffer.add(
        camera=camera_img,
        position=ee_position,
        rotation=ee_rot_axis_angle,
        gripper=gripper_width,
        timestamp=current_time
    )
    #print(f"obs_buffer.get_obs_dict(): {obs_buffer.get_obs_dict()}")
    # 返回格式化的观测字典（包含真实历史）
    return obs_buffer.get_obs_dict()


def apply_action_to_sim(env, action, control_mode="pd_ee_delta_pose"):
    """
    将UMI模型输出的动作应用到仿真环境
    
    Args:
        env: ManiSkill environment
        action: np.ndarray, shape (7,) = [x, y, z, rx, ry, rz, gripper]
        control_mode: control mode string
    """
    # Extract pose and gripper from action
    target_pos = action[:3]
    target_rot_axis_angle = action[3:6]
    target_gripper = action[6]
    
    # Convert to environment action format
    if "delta_pose" in control_mode:
        # For delta pose control, compute delta from current pose
        current_pose = env.agent.tcp.pose
        current_pos = common.to_numpy(current_pose.p).flatten()
        current_quat = common.to_numpy(current_pose.q).flatten()
        current_rot = st.Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        current_rot_axis_angle = current_rot.as_rotvec()
        
        delta_pos = target_pos - current_pos
        delta_rot = target_rot_axis_angle - current_rot_axis_angle
        
        ee_action = np.concatenate([delta_pos, delta_rot])
    else:
        # For absolute pose control
        ee_action = np.concatenate([target_pos, target_rot_axis_angle])
    
    # Map gripper width to gripper action [-1, 1]
    # Assuming gripper_width in [0, 0.09], map to gripper_action
    gripper_action = (target_gripper / 0.045) * 2 - 1  # Map [0, 0.09] to [-1, 1]
    gripper_action = np.clip(gripper_action, -1, 1)
    
    # Create action dict
    action_dict = dict(
        base=np.zeros(2),
        arm=ee_action,
        body=np.zeros(3),
        gripper=gripper_action
    )
    action_dict = common.to_tensor(action_dict)
    action = env.agent.controller.from_action_dict(action_dict)
    
    return action


@click.command()
@click.option('--input', '-i', required=True, help='Path to UMI checkpoint (.ckpt file)')
@click.option('--output', '-o', required=True, help='Directory to save evaluation results')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--env_id', '-e', default='PegInsertionSide-v1', help='ManiSkill environment ID')
@click.option('--obs_mode', default='sensor_data', help='Observation mode')
@click.option('--control_mode', '-c', default='pd_ee_delta_pose', help='Control mode')
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz')
@click.option('--steps_per_inference', '-si', default=6, type=int, help='Action horizon for inference')
@click.option('--enable_viewer', is_flag=True, default=True, help='Enable SAPIEN viewer')
def main(input, output, robot_config, env_id, obs_mode, control_mode, 
         frequency, steps_per_inference, enable_viewer):
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load robot config
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    robots_config = robot_config_data.get('robots', [{}])
    
    # Load UMI checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    
    print(f"Loading checkpoint from: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    print(f"配置文件: cfg: {cfg}")
    
    # Create workspace and load model
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.num_inference_steps = 16  # DDIM inference iterations
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)
    
    print(f"Model loaded on {device}")
    print(f"obs_pose_repr: {obs_pose_rep}")
    print(f"action_pose_repr: {action_pose_repr}")
    
    # Create ManiSkill environment
    print(f"Creating environment: {env_id}")
    env: BaseEnv = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="sensors",
        num_envs=1
    )
    
    obs, _ = env.reset()
    
    if enable_viewer:
        env.render_human()
    
    dt = 1.0 / frequency
    
    print("="*50)
    print("Simulation Ready!")
    print("Controls:")
    print("  - Press 'C' in the visualization window to start policy")
    print("  - Press 'S' to stop policy")
    print("  - Press 'R' to reset environment")
    print("  - Press 'Q' to quit")
    print("="*50)
    
    # Create visualization window
    cv2.namedWindow('Evaluation', cv2.WINDOW_NORMAL)
    
    # 创建观测历史缓冲区（维护真实的时序数据）
    obs_buffer = ObservationBuffer(history_len=2)
    
    policy_active = False
    episode_idx = 0
    
    try:
        while True:
            t_start = time.monotonic()
            
            # Get observation with real history
            obs_dict_np = get_sim_obs_dict(env, obs, obs_buffer)
            
            # Skip if not enough history yet
            if obs_dict_np is None:
                time.sleep(dt)
                continue
            
            # Visualize camera image
            vis_img = (obs_dict_np['camera0_rgb'][-1] * 255).astype(np.uint8)
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            
            status_text = "POLICY ACTIVE" if policy_active else "MANUAL MODE"
            color = (0, 255, 0) if policy_active else (0, 0, 255)
            cv2.putText(vis_img_bgr, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(vis_img_bgr, f"Episode: {episode_idx}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Evaluation', vis_img_bgr)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle keyboard input
            if key == ord('q') or key == ord('Q'):
                print("Quitting...")
                break
            elif key == ord('c') or key == ord('C'):
                print("Starting policy control!")
                policy_active = True
                policy.reset()
                episode_idx += 1
                # Get episode start pose
                episode_start_pose = [np.concatenate([
                    obs_dict_np['robot0_eef_pos'][-1],
                    obs_dict_np['robot0_eef_rot_axis_angle'][-1]
                ])]
            elif key == ord('s') or key == ord('S'):
                print("Stopping policy control!")
                policy_active = False
            elif key == ord('r') or key == ord('R'):
                print("Resetting environment...")
                obs, _ = env.reset()
                obs_buffer.reset()  # 重置观测历史
                policy_active = False
                continue
            
            # Execute action
            if policy_active:
                # Run policy inference
                with torch.no_grad():
                    obs_dict = get_real_umi_obs_dict(
                        env_obs=obs_dict_np,
                        shape_meta=cfg.task.shape_meta,
                        obs_pose_repr=obs_pose_rep,
                        tx_robot1_robot0=np.eye(4),  # Single robot, identity transform
                        episode_start_pose=episode_start_pose
                    )
                    
                    # Debug: print shapes
                    print("\n=== obs_dict shapes ===")
                    for key, val in obs_dict.items():
                        print(f"{key}: {val.shape}")
                    print("======================\n")
                    
                    obs_dict_torch = dict_apply(obs_dict, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    
                    result = policy.predict_action(obs_dict_torch)
                    # 此时得到的是相对坐标
                    # action 是 relative 坐标（10维：pos+rot+gripper）
                    raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                    print(f"raw_action: {raw_action}")
                    
                    # Convert to environment action
                    # 转换后：env_action 是绝对坐标（7维）
                    action_sequence = get_real_umi_action(raw_action, obs_dict_np, action_pose_repr)
                    print(f"action_sequence: {action_sequence}")

                    # Use first action from sequence
                    action_to_execute = action_sequence[0]  # Shape: (7,)
                
                # Apply action to simulation
                sim_action = apply_action_to_sim(env, action_to_execute, control_mode)
                obs, reward, terminated, truncated, info = env.step(sim_action)
                
                if terminated or truncated:
                    print(f"Episode finished! Reward: {reward.item():.3f}")
                    obs, _ = env.reset()
                    obs_buffer.reset()  # 重置观测历史
                    policy_active = False
            else:
                # Manual mode: just step with zero action
                action_dict = dict(
                    base=np.zeros(2),
                    arm=np.zeros(6),
                    body=np.zeros(3),
                    gripper=0
                )
                action_dict = common.to_tensor(action_dict)
                action = env.agent.controller.from_action_dict(action_dict)
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Render
            if enable_viewer:
                env.render_human()
            
            # FPS control
            elapsed = time.monotonic() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    # 保证DP模块在路径中
    sys.path.append("/backup/lerobots/lbg/ManiSkill")
    main()
