import argparse
import signal
import time
import json
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

# We'll handle Ctrl+C gracefully instead of using SIG_DFL
# signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, visualization
from mani_skill.utils.wrappers import RecordEpisode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str, default="sensor_data", help="Observation mode. Use 'sensor_data' to get hand_camera images")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="sensors")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for data collection")
    parser.add_argument("--save-data", action="store_true", help="Save camera images and robot state data")
    parser.add_argument("--data-dir", type=str, default="./collected_data", help="Directory to save collected data")
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        **args.env_kwargs
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, render_mode=args.render_mode)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    obs, _ = env.reset()
    after_reset = True

    # Setup data collection
    target_fps = args.fps
    frame_time = 1.0 / target_fps
    
    if args.save_data:
        # Create data directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path(args.data_dir) / f"session_{timestamp}"
        image_dir = data_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        collected_data = {
            "fps": target_fps,
            "env_id": args.env_id,
            "control_mode": args.control_mode,
            "frames": []
        }
        frame_idx = 0
        print(f"Data will be saved to: {data_dir}")
    
    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    renderer = visualization.ImageRenderer()
    # disable all default plt shortcuts that are lowercase letters
    plt.rcParams["keymap.fullscreen"].remove("f")
    plt.rcParams["keymap.home"].remove("h")
    plt.rcParams["keymap.home"].remove("r")
    plt.rcParams["keymap.back"].remove("c")
    plt.rcParams["keymap.forward"].remove("v")
    plt.rcParams["keymap.pan"].remove("p")
    plt.rcParams["keymap.zoom"].remove("o")
    plt.rcParams["keymap.save"].remove("s")
    plt.rcParams["keymap.grid"].remove("g")
    plt.rcParams["keymap.yscale"].remove("l")
    plt.rcParams["keymap.xscale"].remove("k")

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            env.render_human()
            sapien_viewer = env.viewer
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    gripper_action = 1
    EE_ACTION = 0.1

    try:
        while True:
            # FPS control - start timing
            loop_start_time = time.time()
            
            # -------------------------------------------------------------------------- #
            # Visualization
            # -------------------------------------------------------------------------- #
            if args.enable_sapien_viewer:
                env.render_human()
            
            # Use render_rgb_array() to get image tensor instead of render()
            # which returns a Viewer object when render_mode="human"
            render_frame = env.render_rgb_array().cpu().numpy()[0]
            
            if after_reset:
                after_reset = False
                # Re-focus on opencv viewer
                if args.enable_sapien_viewer:
                    renderer.close()
                    renderer = visualization.ImageRenderer()
                    pass
            # -------------------------------------------------------------------------- #
            # Interaction
            # -------------------------------------------------------------------------- #
            # Input
            renderer(render_frame)
            # key = opencv_viewer.imshow(render_frame.cpu().numpy()[0])
            key = renderer.last_event.key if renderer.last_event is not None else None
            body_action = np.zeros([3])
            base_action = np.zeros([2])  # hardcoded for fetch robot

            # Parse end-effector action
            # 注意: 这里是传入位姿pose
            if (
                "pd_ee_delta_pose" in args.control_mode
                or "pd_ee_target_delta_pose" in args.control_mode
            ):
                ee_action = np.zeros([6])
            # 注意: 这里是传入位置
            elif (
                "pd_ee_delta_pos" in args.control_mode
                or "pd_ee_target_delta_pos" in args.control_mode
            ):
                ee_action = np.zeros([3])
            else:
                raise NotImplementedError(args.control_mode)

            # Base. Hardcoded for Fetch robot at the moment. In the future write interface to do this
            if has_base:
                if key == "w":  # forward
                    base_action[0] = 1
                elif key == "s":  # backward
                    base_action[0] = -1
                elif key == "a":  # rotate counter
                    base_action[1] = 1
                elif key == "d":  # rotate clockwise
                    base_action[1] = -1
                elif key == "z":  # lift
                    body_action[2] = 1
                elif key == "x":  # lower
                    body_action[2] = -1
                elif key == "v":  # rotate head left
                    body_action[0] = 1
                elif key == "b":  # rotate head right
                    body_action[0] = -1
                elif key == "n":  # tilt head down
                    body_action[1] = 1
                elif key == "m":  # rotate head up
                    body_action[1] = -1

            # End-effector
            if num_arms > 0:
                # Position
                if key == "i":  # +x
                    ee_action[0] = EE_ACTION
                elif key == "k":  # -x
                    ee_action[0] = -EE_ACTION
                elif key == "j":  # +y
                    ee_action[1] = EE_ACTION
                elif key == "l":  # -y
                    ee_action[1] = -EE_ACTION
                elif key == "u":  # +z
                    ee_action[2] = EE_ACTION
                elif key == "o":  # -z
                    ee_action[2] = -EE_ACTION

                # Rotation (axis-angle)
                if key == "1":
                    ee_action[3:6] = (1, 0, 0)
                elif key == "2":
                    ee_action[3:6] = (-1, 0, 0)
                elif key == "3":
                    ee_action[3:6] = (0, 1, 0)
                elif key == "4":
                    ee_action[3:6] = (0, -1, 0)
                elif key == "5":
                    ee_action[3:6] = (0, 0, 1)
                elif key == "6":
                    ee_action[3:6] = (0, 0, -1)

            # Gripper
            if has_gripper:
                if key == "f":  # open gripper
                    gripper_action = 1
                elif key == "g":  # close gripper
                    gripper_action = -1

            # Other functions
            if key == "0":  # switch to SAPIEN viewer
                render_wait()
            elif key == "r":  # reset env
                obs, _ = env.reset()
                gripper_action = 1
                after_reset = True
                continue
            elif key == None:  # exit
                break

            # Visualize observation
            if key == "v":
                if "pointcloud" in env.obs_mode:
                    import trimesh

                    xyzw = obs["pointcloud"]["xyzw"]
                    mask = xyzw[..., 3] > 0
                    rgb = obs["pointcloud"]["rgb"]
                    if "robot_seg" in obs["pointcloud"]:
                        robot_seg = obs["pointcloud"]["robot_seg"]
                        rgb = np.uint8(robot_seg * [11, 61, 127])
                    trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()

            # -------------------------------------------------------------------------- #
            # Post-process action
            # -------------------------------------------------------------------------- #
            action_dict = dict(
                base=base_action, arm=ee_action, body=body_action, gripper=gripper_action
            )
            action_dict = common.to_tensor(action_dict)
            action = env.agent.controller.from_action_dict(action_dict)

            obs, reward, terminated, truncated, info = env.step(action)
            print("reward", reward)
            print("terminated", terminated, "truncated", truncated)
            print("info", info)
        
            # -------------------------------------------------------------------------- #
            # Data Collection (30 FPS)
            # -------------------------------------------------------------------------- #
            if args.save_data:
                # Get end-effector pose
                # env.agent.tcp is the tool center point (end-effector)
                ee_pose = env.agent.tcp.pose  # Returns sapien.Pose object
                ee_position = ee_pose.p[0]  # xyz position, [0] removes batch dimension: (1, 3) -> (3,)
                ee_quaternion = ee_pose.q[0]  # quaternion [w, x, y, z], [0] removes batch dimension: (1, 4) -> (4,)
                
                # Get gripper state (qpos = joint positions)
                # The exact gripper joints depend on the robot, typically last few joints
                qpos = env.agent.robot.get_qpos()  # Shape: (1, 9) for single env with 9 joints
                qpos = qpos[0]  # Remove batch dimension: (1, 9) -> (9,)
                gripper_qpos = qpos[-2:] if len(qpos) > 2 else qpos  # Last 2 joints are gripper joints
                
                # Get gripper action value
                gripper_width = gripper_action  # Current gripper command
                
                # Get camera images from observation sensor_data
                # obs["sensor_data"] contains images from all cameras
                camera_images = {}
                if "sensor_data" in obs:
                    # Debug: Print sensor data structure (uncomment if needed for debugging)
                    # print("==== obs['sensor_data'] ====")
                    # for cam_name, cam_data in obs["sensor_data"].items():
                    #     print(f"[{cam_name}]")
                    #     if isinstance(cam_data, dict):
                    #         for k, v in cam_data.items():
                    #             if hasattr(v, 'shape'):
                    #                 print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    
                    for cam_name, cam_data in obs["sensor_data"].items():
                        # Check for Color (RGBA format from ManiSkill sensor)
                        # 只保存hand_camera的图像
                        if cam_name == "hand_camera" and "Color" in cam_data:
                            # Get Color image from camera (RGBA format, torch tensor)
                            color_img = common.to_numpy(cam_data["Color"][0])  # [0] for first env, shape: (H, W, 4)
                            
                            # Convert RGBA to RGB by dropping the alpha channel
                            rgb_img = color_img[:, :, :3]  # Take only RGB channels
                            camera_images[cam_name] = rgb_img
                            
                            # Save each camera's image
                            cam_image_path = image_dir / f"frame_{frame_idx:06d}_{cam_name}.png"
                            from PIL import Image
                            
                            # Data is already uint8 from sensor
                            if rgb_img.dtype == np.uint8:
                                img_to_save = rgb_img
                            elif rgb_img.max() <= 1.0:
                                img_to_save = (rgb_img * 255).astype(np.uint8)
                            else:
                                img_to_save = rgb_img.astype(np.uint8)
                            
                            Image.fromarray(img_to_save).save(cam_image_path)
                
                # 不用再保存 Also save the render frame for visualization
                # render_image_path = image_dir / f"frame_{frame_idx:06d}_render.png"
                # from PIL import Image
                # img_to_save = (render_frame * 255).astype(np.uint8) if render_frame.max() <= 1.0 else render_frame.astype(np.uint8)
                # Image.fromarray(img_to_save).save(render_image_path)
                
                print(f"debug ee_quaternion: {ee_quaternion}")

                # Collect frame data
                frame_data = {
                    "frame_idx": frame_idx,
                    "timestamp": time.time(),
                    "render_image": f"frame_{frame_idx:06d}_render.png",
                    "camera_images": {cam_name: f"frame_{frame_idx:06d}_{cam_name}.png" for cam_name in camera_images.keys()},
                    "end_effector": {
                        "position": ee_position.tolist(),  # [x, y, z]
                        "quaternion": ee_quaternion.tolist()  # [w, x, y, z]
                    },
                    "gripper": {
                        "qpos": gripper_qpos.tolist(),  # Only gripper joints (last 2)
                        "action": float(gripper_width),
                    },
                    "robot_qpos": qpos.tolist(),  # Full joint positions (all 9 joints)
                    "action": {
                        "ee_action": ee_action.tolist(),
                        "base_action": base_action.tolist(),
                        "body_action": body_action.tolist(),
                        "gripper_action": float(gripper_action),
                    },
                    "reward": float(reward),
                }
                
                collected_data["frames"].append(frame_data)
                frame_idx += 1
                
                # Print data collection info
                print(f"[Frame {frame_idx}] EE Pos: {ee_position}, Gripper: {gripper_qpos}")
        
            # -------------------------------------------------------------------------- #
            # FPS Control - sleep to maintain target FPS
            # -------------------------------------------------------------------------- #
            loop_elapsed = time.time() - loop_start_time
            sleep_time = frame_time - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            actual_fps = 1.0 / (time.time() - loop_start_time)
            if args.save_data and frame_idx % 30 == 0:  # Print every 30 frames
                print(f"Actual FPS: {actual_fps:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nReceived Ctrl+C, saving data before exit...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}, saving data before exit...")
    finally:
        # Save collected data to JSON file (always executed, even on Ctrl+C)
        if args.save_data:
            json_path = data_dir / "collected_data.json"
            with open(json_path, 'w') as f:
                json.dump(collected_data, f, indent=2)
            print(f"\nData saved to {data_dir}")
            print(f"Total frames collected: {frame_idx}")
            print(f"Images saved to: {image_dir}")
            print(f"Metadata saved to: {json_path}")

        env.close()


if __name__ == "__main__":
    main()