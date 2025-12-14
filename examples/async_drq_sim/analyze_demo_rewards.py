import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_rewards(pkl_path):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        # Try looking in current directory if path includes directory prefix
        if pkl_path.name == pkl_path.as_posix():
             pass # already just a filename
        elif Path(pkl_path.name).exists():
             pkl_path = Path(pkl_path.name)
        else:
             print(f"Error: {pkl_path} not found")
             return

    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)

    print(f"Loaded {len(transitions)} transitions")

    # Infer episodes based on reward drops (reset)
    # Since the pkl has one long episode, we manually split when reward drops significantly
    # e.g. from >0.1 to <0.01
    
    inferred_episodes = []
    current_ep_rewards = []
    
    # Extract raw rewards first
    raw_rewards = [t['rewards'] for t in transitions]
    
    for r in raw_rewards:
        # If we are in an episode (len > 0) and reward drops to near zero
        # it usually means a reset happened.
        if len(current_ep_rewards) > 0 and r < 0.01 and current_ep_rewards[-1] > 0.1:
            inferred_episodes.append(current_ep_rewards)
            current_ep_rewards = []
        
        current_ep_rewards.append(r)
        
    if current_ep_rewards:
        inferred_episodes.append(current_ep_rewards)
        
    print(f"\nInferred {len(inferred_episodes)} episodes based on reward resets.")
    
    print("\n--- Detailed Episode Analysis ---")
    all_max_rewards = []
    
    for i, ep_rewards in enumerate(inferred_episodes):
        ep_rewards = np.array(ep_rewards)
        max_r = np.max(ep_rewards)
        mean_r = np.mean(ep_rewards)
        final_r = ep_rewards[-1]
        
        # Analyze the stable hold phase (e.g., last 10 steps BEFORE reset or end)
        # Often the last few steps are the reset (0.0), so we look at the peak
        # or the steps right before the drop.
        
        # Find steps where reward > 0.5 (lifting phase)
        high_rewards = ep_rewards[ep_rewards > 0.5]
        if len(high_rewards) > 0:
            avg_high = np.mean(high_rewards)
            min_high = np.min(high_rewards)
        else:
            avg_high = 0.0
            min_high = 0.0
            
        all_max_rewards.append(max_r)
        
        print(f"Ep {i+1:02d}: Steps={len(ep_rewards):<3} | Max={max_r:.4f} | Avg(High)={avg_high:.4f} | Min(High)={min_high:.4f}")
        
        # Print ALL rewards for ALL episodes (Commented out to reduce spam)
        # print(f"\n--- Detailed Rewards for Episode {i+1} ---")
        # for step_idx, r in enumerate(ep_rewards):
        #     print(f"Step {step_idx}: {r:.4f}")
        # print("--------------------------------------\n")

    # Determine threshold suggestions
    success_rewards = np.array(all_max_rewards)
    
    # Let's see the distribution of "high" rewards
    p10 = np.percentile(success_rewards, 10)
    p50 = np.percentile(success_rewards, 50)
    p90 = np.percentile(success_rewards, 90)

    print("\n--- Threshold Suggestions ---")
    print(f"10th percentile of Max Rewards in demos: {p10:.4f}")
    print(f"Median of Max Rewards in demos: {p50:.4f}")
    print(f"Min of Max Rewards in demos: {np.min(success_rewards):.4f}")
    
    print(f"\nSuggestion: Since these are expert demos, the agent is considered 'successful' if it reaches a reward similar to these demos.")
    
    # Also check what the reward looks like when it's STABLE (e.g. holding the block)
    # We can look at the last 10 steps of each episode
    stable_rewards = []
    for ep in inferred_episodes:
        if len(ep) > 10:
            stable_rewards.extend(ep[-10:])
    
    if stable_rewards:
        min_stable = np.min(stable_rewards)
        avg_stable = np.mean(stable_rewards)
        print(f"\n--- Stable Holding Analysis (Last 10 steps) ---")
        print(f"Min reward during stable hold: {min_stable:.4f}")
        print(f"Avg reward during stable hold: {avg_stable:.4f}")
        print(f"Proposed Success Threshold: {min_stable - 0.05:.2f}")

if __name__ == "__main__":
    analyze_rewards("examples/async_drq_sim/franka_lift_cube_image_20_trajs.pkl")

