import pickle
from pathlib import Path
import numpy as np

def convert_to_sparse(input_path, output_path, threshold=0.9):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        # Try looking in current directory if path includes directory prefix
        if Path(input_path.name).exists():
             input_path = Path(input_path.name)
             # Update output path to be in current directory too if input was found here
             if str(output_path).startswith(str(input_path.parent)):
                 output_path = Path(output_path.name)
        else:
            print(f"Error: {input_path} not found")
            return

    print(f"Loading transitions from {input_path}...")
    with open(input_path, "rb") as f:
        transitions = pickle.load(f)

    print(f"Loaded {len(transitions)} transitions")
    
    success_count = 0
    modified_transitions = []
    
    for i, t in enumerate(transitions):
        original_reward = t['rewards']
        
        # Apply sparse logic
        if original_reward > threshold:
            new_reward = 1.0
            success_count += 1
        else:
            new_reward = 0.0
            
        # Create a new dict to avoid modifying the original list in place if referenced elsewhere
        # (Though here we are just processing linearly)
        new_t = t.copy()
        new_t['rewards'] = new_reward
        modified_transitions.append(new_t)

    print(f"Conversion complete.")
    print(f"Total transitions: {len(modified_transitions)}")
    print(f"Success frames (reward=1.0): {success_count} ({success_count/len(modified_transitions)*100:.2f}%)")
    print(f"Saving to {output_path}...")
    
    with open(output_path, "wb") as f:
        pickle.dump(modified_transitions, f)
    
    print("Done!")

if __name__ == "__main__":
    # Use simple filenames. The script logic above will handle finding them in current dir.
    input_file = "franka_lift_cube_image_20_trajs.pkl" 
    output_file = "franka_lift_cube_image_20_trajs_sparse.pkl"
    
    # If running from root, prepend the directory
    if Path("examples/async_drq_sim").exists():
        input_file = "examples/async_drq_sim/" + input_file
        output_file = "examples/async_drq_sim/" + output_file
        
    convert_to_sparse(input_file, output_file, threshold=0.8)

