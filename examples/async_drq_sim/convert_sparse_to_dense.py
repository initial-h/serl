import pickle
import numpy as np
from pathlib import Path
import copy

def convert_sparse_to_dense(input_path, output_path, discount=0.8):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    with open(input_path, "rb") as f:
        transitions = pickle.load(f)

    print(f"Loaded {len(transitions)} transitions from {input_path}")

    # Group into episodes
    episodes = []
    current_episode = []
    
    for i, t in enumerate(transitions):
        r = t['rewards']
        
        # Check for reset: drop from success (1.0) to start (0.0)
        # Using thresholds 0.9 and 0.1 to be safe, though sparse is usually exact
        if len(current_episode) > 0 and r < 0.1 and current_episode[-1]['rewards'] > 0.9:
            episodes.append(current_episode)
            current_episode = []
        
        current_episode.append(t)
    
    if current_episode:
        episodes.append(current_episode)
        
    print(f"Inferred {len(episodes)} episodes.")

    new_transitions = []
    
    for ep_idx, episode in enumerate(episodes):
        # Create a deep copy to avoid modifying original if we needed it (though we just write new list)
        # Actually modifying the dicts in the list is fine if we output a new list of dicts.
        # But we need to keep the structure.
        
        # Find first success step
        first_success_idx = -1
        for i, t in enumerate(episode):
            if t['rewards'] >= 0.99: # approximate 1.0
                first_success_idx = i
                break
        
        if first_success_idx != -1:
            print(f"Ep {ep_idx+1}: First success at step {first_success_idx}. Applying discount {discount} backwards.")
            
            # Apply backwards decay
            # reward at first_success_idx is 1.0
            # reward at first_success_idx - 1 is 1.0 * discount
            # reward at first_success_idx - 2 is 1.0 * discount^2
            # ...
            # reward at 0 is 1.0 * discount^(first_success_idx)
            
            for i in range(first_success_idx):
                steps_back = first_success_idx - i
                decayed_reward = 1.0 * (discount ** steps_back)
                episode[i]['rewards'] = decayed_reward
        else:
            print(f"Ep {ep_idx+1}: No success found. Skipping dense reward generation.")
            
        new_transitions.extend(episode)

    print(f"Saving {len(new_transitions)} dense transitions to {output_path}")
    
    with open(output_path, "wb") as f:
        pickle.dump(new_transitions, f)
        
    print("Done.")

if __name__ == "__main__":
    convert_sparse_to_dense(
        "examples/async_drq_sim/franka_lift_cube_image_20_trajs_sparse.pkl",
        "examples/async_drq_sim/franka_lift_cube_image_20_trajs_dense_generated.pkl",
        discount=0.8
    )

