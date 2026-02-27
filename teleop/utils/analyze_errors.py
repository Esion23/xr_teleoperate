import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
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

    # Helper to convert list of lists to numpy array
    def extract_arr(key):
        return np.array([d[key] for d in data])

    # Extract Raw Data
    timestamps = extract_arr('timestamp') if 'timestamp' in data[0] else np.arange(len(data)) * 0.033
    if 'timestamp' in data[0]:
        timestamps = timestamps - timestamps[0] # Relative time

    target_l_pos = extract_arr('target_l_pos')
    target_r_pos = extract_arr('target_r_pos')
    target_l_rot = extract_arr('target_l_rot')
    target_r_rot = extract_arr('target_r_rot')
    
    actual_l_pos = extract_arr('actual_l_pos')
    actual_r_pos = extract_arr('actual_r_pos')
    actual_l_rot = extract_arr('actual_l_rot')
    actual_r_rot = extract_arr('actual_r_rot')
    
    sol_l_pos = extract_arr('sol_l_pos')
    sol_r_pos = extract_arr('sol_r_pos')
    sol_l_rot = extract_arr('sol_l_rot')
    sol_r_rot = extract_arr('sol_r_rot')
    
    cmd_joints = extract_arr('cmd_joints')
    actual_joints = extract_arr('actual_joints')
    ik_time = extract_arr('ik_time')

    # --- Calculate Errors (Offline) ---
    N = len(data)
    
    # 1. Tracking Error (Steady-state: Target[t-1] vs Actual[t])
    track_pos_err_l = np.zeros(N)
    track_pos_err_r = np.zeros(N)
    track_rot_err_l = np.zeros(N)
    track_rot_err_r = np.zeros(N)
    
    for i in range(1, N):
        # Position Error
        track_pos_err_l[i] = np.linalg.norm(target_l_pos[i-1] - actual_l_pos[i])
        track_pos_err_r[i] = np.linalg.norm(target_r_pos[i-1] - actual_r_pos[i])
        
        # Rotation Error (Geodesic)
        R_diff_l = target_l_rot[i-1] @ actual_l_rot[i].T
        track_rot_err_l[i] = np.arccos(np.clip((np.trace(R_diff_l) - 1) / 2, -1.0, 1.0))
        
        R_diff_r = target_r_rot[i-1] @ actual_r_rot[i].T
        track_rot_err_r[i] = np.arccos(np.clip((np.trace(R_diff_r) - 1) / 2, -1.0, 1.0))

    # 2. Solver Error (Target[t] vs Solution[t])
    solve_pos_err_l = np.zeros(N)
    solve_pos_err_r = np.zeros(N)
    solve_rot_err_l = np.zeros(N)
    solve_rot_err_r = np.zeros(N)

    for i in range(N):
        solve_pos_err_l[i] = np.linalg.norm(target_l_pos[i] - sol_l_pos[i])
        solve_pos_err_r[i] = np.linalg.norm(target_r_pos[i] - sol_r_pos[i])
        
        R_diff_l = target_l_rot[i] @ sol_l_rot[i].T
        solve_rot_err_l[i] = np.arccos(np.clip((np.trace(R_diff_l) - 1) / 2, -1.0, 1.0))
        
        R_diff_r = target_r_rot[i] @ sol_r_rot[i].T
        solve_rot_err_r[i] = np.arccos(np.clip((np.trace(R_diff_r) - 1) / 2, -1.0, 1.0))

    # 3. Execution Error (Cmd[t-1] vs Actual[t])
    exec_err = np.zeros(N)
    for i in range(1, N):
        exec_err[i] = np.linalg.norm(cmd_joints[i-1] - actual_joints[i])

    # --- Statistical Analysis ---
    metrics = {
        'Track Pos Err L': track_pos_err_l, 'Track Pos Err R': track_pos_err_r,
        'Track Rot Err L': track_rot_err_l, 'Track Rot Err R': track_rot_err_r,
        'Solve Pos Err L': solve_pos_err_l, 'Solve Pos Err R': solve_pos_err_r,
        'Exec Err': exec_err
    }
    
    print(f"\n{'Metric':<25} | {'Mean':<10} | {'Max':<10} | {'Std Dev':<10}")
    print("-" * 65)
    for name, values in metrics.items():
        print(f"{name:<25} | {np.mean(values):.4f}     | {np.max(values):.4f}     | {np.std(values):.4f}")

    # --- Figure 1: Error Analysis Plots (Static) ---
    fig_err, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig_err.suptitle(f'Error Analysis: {os.path.basename(log_file)}', fontsize=16)

    # 1. Position Errors (Left/Right)
    axes[0, 0].plot(timestamps, track_pos_err_l, label='Tracking (Target-Actual)', alpha=0.7)
    axes[0, 0].plot(timestamps, solve_pos_err_l, label='Solver (Target-IK)', alpha=0.7)
    axes[0, 0].set_title('Left Hand Position Error (m)')
    axes[0, 0].set_ylabel('Error (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(timestamps, track_pos_err_r, label='Tracking (Target-Actual)', alpha=0.7)
    axes[0, 1].plot(timestamps, solve_pos_err_r, label='Solver (Target-IK)', alpha=0.7)
    axes[0, 1].set_title('Right Hand Position Error (m)')
    axes[0, 1].grid(True)

    # 2. Rotation Errors (Left/Right)
    axes[1, 0].plot(timestamps, track_rot_err_l, label='Tracking (Target-Actual)', color='orange', alpha=0.7)
    axes[1, 0].plot(timestamps, solve_rot_err_l, label='Solver (Target-IK)', color='green', alpha=0.7)
    axes[1, 0].set_title('Left Hand Rotation Error (rad)')
    axes[1, 0].set_ylabel('Error (rad)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(timestamps, track_rot_err_r, label='Tracking (Target-Actual)', color='orange', alpha=0.7)
    axes[1, 1].plot(timestamps, solve_rot_err_r, label='Solver (Target-IK)', color='green', alpha=0.7)
    axes[1, 1].set_title('Right Hand Rotation Error (rad)')
    axes[1, 1].grid(True)

    # 3. Execution & IK Time
    axes[2, 0].plot(timestamps, exec_err, color='purple', alpha=0.8)
    axes[2, 0].set_title('Execution Error (Joint Space Norm)')
    axes[2, 0].set_ylabel('Norm')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True)

    axes[2, 1].plot(timestamps, ik_time * 1000, color='brown', alpha=0.8) # Convert to ms
    axes[2, 1].set_title('IK Solve Time (ms)')
    axes[2, 1].set_ylabel('Time (ms)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True)
    axes[2, 1].axhline(y=33.3, color='r', linestyle='--', alpha=0.5, label='33ms (30Hz)')
    axes[2, 1].legend()

    plt.tight_layout()
    
    # Save Error Plot
    output_img = log_file.replace('.jsonl', '_errors.png')
    fig_err.savefig(output_img)
    print(f"\nError analysis plot saved to: {output_img}")

    # --- Figure 2: Interactive 3D Trajectory ---
    fig_3d = plt.figure(figsize=(16, 8))
    fig_3d.suptitle(f'3D Trajectory Replay: {os.path.basename(log_file)}', fontsize=16)

    # 3D Plot for Left Hand
    ax1 = fig_3d.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title('Left Hand Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 3D Plot for Right Hand
    ax2 = fig_3d.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title('Right Hand Trajectory')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Initialize plot elements
    # Full Trajectory (faint)
    # Align trajectories for visualization: Target[0:-1] vs Actual[1:]
    viz_target_l = target_l_pos[:-1]
    viz_actual_l = actual_l_pos[1:]
    viz_target_r = target_r_pos[:-1]
    viz_actual_r = actual_r_pos[1:]
    
    ax1.plot(viz_target_l[:, 0], viz_target_l[:, 1], viz_target_l[:, 2], 'g--', alpha=0.1)
    ax1.plot(viz_actual_l[:, 0], viz_actual_l[:, 1], viz_actual_l[:, 2], 'b-', alpha=0.1)
    ax2.plot(viz_target_r[:, 0], viz_target_r[:, 1], viz_target_r[:, 2], 'g--', alpha=0.1)
    ax2.plot(viz_actual_r[:, 0], viz_actual_r[:, 1], viz_actual_r[:, 2], 'r-', alpha=0.1)

    # Current Frame Markers
    l_target_pt, = ax1.plot([], [], [], 'go', markersize=8, label='Target[t-1]')
    l_actual_pt, = ax1.plot([], [], [], 'bo', markersize=8, label='Actual[t]')
    l_trail_target, = ax1.plot([], [], [], 'g--', alpha=0.5) # Trail
    l_trail_actual, = ax1.plot([], [], [], 'b-', alpha=0.5)

    r_target_pt, = ax2.plot([], [], [], 'go', markersize=8, label='Target[t-1]')
    r_actual_pt, = ax2.plot([], [], [], 'ro', markersize=8, label='Actual[t]')
    r_trail_target, = ax2.plot([], [], [], 'g--', alpha=0.5)
    r_trail_actual, = ax2.plot([], [], [], 'r-', alpha=0.5)

    ax1.legend()
    ax2.legend()

    # Set axis limits
    def set_axes_equal(ax, x, y, z):
        x_limits = [np.min(x), np.max(x)]
        y_limits = [np.min(y), np.max(y)]
        z_limits = [np.min(z), np.max(z)]
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax1, viz_target_l[:,0], viz_target_l[:,1], viz_target_l[:,2])
    set_axes_equal(ax2, viz_target_r[:,0], viz_target_r[:,1], viz_target_r[:,2])

    # Slider for time control
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(viz_target_l)-1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        
        # We are visualizing frame i, which means:
        # Target at index i (which is originally i) -> No, wait.
        # viz_target[i] is target[i]
        # viz_actual[i] is actual[i+1]
        
        # Update Left Hand
        l_target_pt.set_data([viz_target_l[idx, 0]], [viz_target_l[idx, 1]])
        l_target_pt.set_3d_properties([viz_target_l[idx, 2]])
        
        l_actual_pt.set_data([viz_actual_l[idx, 0]], [viz_actual_l[idx, 1]])
        l_actual_pt.set_3d_properties([viz_actual_l[idx, 2]])

        # Trail (last 50 frames)
        start = max(0, idx - 50)
        l_trail_target.set_data(viz_target_l[start:idx+1, 0], viz_target_l[start:idx+1, 1])
        l_trail_target.set_3d_properties(viz_target_l[start:idx+1, 2])
        l_trail_actual.set_data(viz_actual_l[start:idx+1, 0], viz_actual_l[start:idx+1, 1])
        l_trail_actual.set_3d_properties(viz_actual_l[start:idx+1, 2])

        # Update Right Hand
        r_target_pt.set_data([viz_target_r[idx, 0]], [viz_target_r[idx, 1]])
        r_target_pt.set_3d_properties([viz_target_r[idx, 2]])
        
        r_actual_pt.set_data([viz_actual_r[idx, 0]], [viz_actual_r[idx, 1]])
        r_actual_pt.set_3d_properties([viz_actual_r[idx, 2]])

        # Trail
        r_trail_target.set_data(viz_target_r[start:idx+1, 0], viz_target_r[start:idx+1, 1])
        r_trail_target.set_3d_properties(viz_target_r[start:idx+1, 2])
        r_trail_actual.set_data(viz_actual_r[start:idx+1, 0], viz_actual_r[start:idx+1, 1])
        r_trail_actual.set_3d_properties(viz_actual_r[start:idx+1, 2])
        
        fig_3d.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Auto Play
    is_playing = False
    def play(event):
        nonlocal is_playing
        if is_playing:
            return
        is_playing = True
        for i in range(int(slider.val), len(viz_target_l)):
            if not plt.fignum_exists(fig_3d.number): break
            slider.set_val(i)
            plt.pause(0.01)
        is_playing = False

    ax_play = plt.axes([0.85, 0.05, 0.1, 0.04])
    btn_play = Button(ax_play, 'Play')
    btn_play.on_clicked(play)

    print("Interactive plot opened. Use slider or 'Play' button. Rotate view with mouse.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajectory replay")
    parser.add_argument("file", type=str, help="Path to the .jsonl log file")
    args = parser.parse_args()
    
    analyze_error_log(args.file)
