import sys
import os
import numpy as np

# Add project root and franka_sim to path
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'franka_sim'))

try:
    from franka_sim.franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
except ImportError:
    # Fallback import if the structure is slightly different
    from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv

def run_oracle_test():
    env = PandaPickCubeGymEnv(render_mode="rgb_array") # use rgb_array to avoid window popup
    obs, _ = env.reset()
    
    total_reward = 0
    # Calculate max steps from environment definition
    # time_limit is stored as _time_limit in MujocoGymEnv
    time_limit = env._time_limit
    max_steps = int(time_limit / env.control_dt)
    print(f"Environment Definition: Time Limit={time_limit}s, Control dt={env.control_dt}s => Max Steps={max_steps}")
    
    print(f"Checking reward for {max_steps} steps...")
    
    for i in range(max_steps):
        # --- CHEAT MODE START ---
        # 1. 强制将方块瞬移到夹爪中心 (满足 r_close = 1.0)
        tcp_pos = env.data.sensor("2f85/pinch_pos").data
        env.model.body("block").pos = tcp_pos  # 注意：这里可能需要调整 offset，取决于坐标系定义
        # 或者更直接：直接修改 qpos
        # block joint qpos: [x, y, z, qw, qx, qy, qz]
        # 但是直接改 model.body.pos 可能无效，通常需要改 data.qpos 或 data.mocap
        
        # 让我们直接通过 env._data 访问内部数据来瞬移方块
        # 获取夹爪当前位置
        gripper_pos = env._data.sensor("2f85/pinch_pos").data.copy()
        
        # 将方块位置设置为夹爪位置 (r_close max)
        # block joint 的 qpos 前3位是 x,y,z
        # 注意：这里需要知道 block joint 的名字或者 id
        block_joint_id = env._model.joint("block").id
        block_qpos_adr = env._model.jnt_qposadr[block_joint_id]
        
        env._data.qpos[block_qpos_adr : block_qpos_adr+3] = gripper_pos
        
        # 2. 强制将方块和夹爪都抬高到目标高度 (满足 r_lift = 1.0)
        # 目标高度是 env._z_success
        target_z = env._z_success + 0.05 #稍微高一点点确保超过阈值
        
        # 修改方块高度
        env._data.qpos[block_qpos_adr + 2] = target_z
        
        # 修改夹爪(mocap)高度，否则 r_close 又变小了
        # mocap_pos 控制夹爪位置
        current_mocap_pos = env._data.mocap_pos[0].copy()
        current_mocap_pos[2] = target_z 
        # 还要把 x,y 也对齐，不然还是有距离
        current_mocap_pos[0] = env._data.qpos[block_qpos_adr]
        current_mocap_pos[1] = env._data.qpos[block_qpos_adr + 1]
        env._data.mocap_pos[0] = current_mocap_pos
        
        # 必须调用 forward 让修改生效
        import mujoco
        mujoco.mj_forward(env._model, env._data)
        
        # --- CHEAT MODE END ---
        
        # Step environment (action doesn't matter much as we force state, but keep gripper closed)
        # Action: [x, y, z, gripper] -> gripper=1 means close? No, usually -1 or 1 depending on setup.
        # In this env: 
        #   grasp = action[3]
        #   dg = grasp * scale
        #   ng = clip(g + dg, 0, 1)
        # Let's try to close it
        obs, rew, done, truncated, info = env.step(np.array([0, 0, 0, 1.0]))
        
        total_reward += rew
        
        # Debug print for first few steps to ensure cheat works
        if i < 100:
            # Re-calculate parts manually to verify
            block_pos = env._data.sensor("block_pos").data
            dist = np.linalg.norm(block_pos - env._data.sensor("2f85/pinch_pos").data)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - env._z_init) / (env._z_success - env._z_init)
            print(f"Step {i}: Reward={rew:.4f} (Dist={dist:.4f}, Lift={r_lift:.4f})")

        
    print(f"Theoretical Max Undiscounted Return over {max_steps} steps: {total_reward:.2f}")
    
    # Now let's see what the environment returns if we just stand still (lower bound)
    env.reset()
    total_actual_reward = 0
    for i in range(max_steps):
        obs, rew, done, truncated, info = env.step(np.zeros(4))
        total_actual_reward += rew
        
    print(f"Lower Bound Return (Do Nothing) over {max_steps} steps: {total_actual_reward:.2f}")

    # Explicitly close the environment to avoid context destruction errors
    env.close()

if __name__ == "__main__":
    run_oracle_test()

