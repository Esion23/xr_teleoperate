import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def analyze_error_log(log_file):
    if not os.path.exists(log_file):
        print(f"Error: File {log_file} not found.")
        return

    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not data:
        print("No valid data found in log file.")
        return

    # Convert to numpy structure for easier slicing
    timestamps = np.array([d['timestamp'] for d in data])
    timestamps = timestamps - timestamps[0] # Relative time

    metrics = {
        'track_pos_err_l': np.array([d['track_pos_err_l'] for d in data]),
        'track_pos_err_r': np.array([d['track_pos_err_r'] for d in data]),
        'track_rot_err_l': np.array([d['track_rot_err_l'] for d in data]),
        'track_rot_err_r': np.array([d['track_rot_err_r'] for d in data]),
        'solve_pos_err_l': np.array([d['solve_pos_err_l'] for d in data]),
        'solve_pos_err_r': np.array([d['solve_pos_err_r'] for d in data]),
        'solve_rot_err_l': np.array([d['solve_rot_err_l'] for d in data]),
        'solve_rot_err_r': np.array([d['solve_rot_err_r'] for d in data]),
        'exec_err': np.array([d['exec_err'] for d in data]),
        'ik_time': np.array([d['ik_time'] for d in data])
    }

    # --- Statistical Analysis ---
    print(f"{'Metric':<25} | {'Mean':<10} | {'Max':<10} | {'Std Dev':<10}")
    print("-" * 65)
    for name, values in metrics.items():
        print(f"{name:<25} | {np.mean(values):.4f}     | {np.max(values):.4f}     | {np.std(values):.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Error Analysis: {os.path.basename(log_file)}', fontsize=16)

    # 1. Position Errors (Left/Right)
    axes[0, 0].plot(timestamps, metrics['track_pos_err_l'], label='Tracking (Target-Actual)', alpha=0.7)
    axes[0, 0].plot(timestamps, metrics['solve_pos_err_l'], label='Solver (Target-IK)', alpha=0.7)
    axes[0, 0].set_title('Left Hand Position Error (m)')
    axes[0, 0].set_ylabel('Error (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(timestamps, metrics['track_pos_err_r'], label='Tracking (Target-Actual)', alpha=0.7)
    axes[0, 1].plot(timestamps, metrics['solve_pos_err_r'], label='Solver (Target-IK)', alpha=0.7)
    axes[0, 1].set_title('Right Hand Position Error (m)')
    axes[0, 1].grid(True)

    # 2. Rotation Errors (Left/Right)
    axes[1, 0].plot(timestamps, metrics['track_rot_err_l'], label='Tracking (Target-Actual)', color='orange', alpha=0.7)
    axes[1, 0].plot(timestamps, metrics['solve_rot_err_l'], label='Solver (Target-IK)', color='green', alpha=0.7)
    axes[1, 0].set_title('Left Hand Rotation Error (rad)')
    axes[1, 0].set_ylabel('Error (rad)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(timestamps, metrics['track_rot_err_r'], label='Tracking (Target-Actual)', color='orange', alpha=0.7)
    axes[1, 1].plot(timestamps, metrics['solve_rot_err_r'], label='Solver (Target-IK)', color='green', alpha=0.7)
    axes[1, 1].set_title('Right Hand Rotation Error (rad)')
    axes[1, 1].grid(True)

    # 3. Execution & IK Time
    axes[2, 0].plot(timestamps, metrics['exec_err'], color='purple', alpha=0.8)
    axes[2, 0].set_title('Execution Error (Joint Space Norm)')
    axes[2, 0].set_ylabel('Norm')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True)

    axes[2, 1].plot(timestamps, metrics['ik_time'] * 1000, color='brown', alpha=0.8) # Convert to ms
    axes[2, 1].set_title('IK Solve Time (ms)')
    axes[2, 1].set_ylabel('Time (ms)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True)
    
    # Add horizontal line for 33ms (30Hz) reference
    axes[2, 1].axhline(y=33.3, color='r', linestyle='--', alpha=0.5, label='33ms (30Hz)')
    axes[2, 1].legend()

    plt.tight_layout()
    
    output_img = log_file.replace('.jsonl', '.png')
    plt.savefig(output_img)
    print(f"\nAnalysis plot saved to: {output_img}")
    # plt.show() # Uncomment if running in an environment with display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze error logs from xr_teleoperate")
    parser.add_argument("file", type=str, help="Path to the .jsonl log file")
    args = parser.parse_args()
    
    analyze_error_log(args.file)
