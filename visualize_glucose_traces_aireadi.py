import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from improved_simulator import ImprovedGlucoseInsulinSimulator
from pid_controller import PIDController
from dqn_agent import DQNAgent, ReplayBuffer
import torch

# Load data
df = pd.read_csv('aireadi_cgm_kcal.csv', parse_dates=['timestamp'])
subjects = df['ID'].unique()

high_glucose_ids = pd.read_csv('high_glucose_aireadi.csv')['participant_id'].unique()
subjects = high_glucose_ids

print(subjects)
print(high_glucose_ids)

print(type(subjects))
print(type(subjects))

# Optimal parameters
OPTIMAL_BOLUS_RATIO = 1/5000
OPTIMAL_KP = 0.005
OPTIMAL_KI = 0.0001
OPTIMAL_KD = 0.005
OPTIMAL_BASAL = 0.05

# Load trained RL agent
HIST_LEN = 5
state_dim = HIST_LEN + 1
action_dim = 13
trained_agent = DQNAgent(state_dim, action_dim)
trained_agent.q_net.load_state_dict(torch.load('aireadi/rl_agent_simplified_v2.pth'))
trained_agent.epsilon = 0.01

# Action space for RL
action_space = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

print("Creating comprehensive glucose trace visualizations...")

# Select top performing subjects for detailed visualization
top_subjects = [1001, 1002, 1003]  # Best RL performers

top_subjects = subjects

for subject_id in top_subjects:
    print(f"\nProcessing subject {subject_id}...")
    
    # Get subject data
    sub = df[df['ID'] == subject_id].sort_values('timestamp').reset_index(drop=True)
    kcal = sub['kcal'].values
    original_glucose = sub['cgm'].values
    timestamps = sub['timestamp'].values

    # Run controllers
    initial_glucose = sub['cgm'].iloc[0]
    
    # PID Controller
    sim_pid = ImprovedGlucoseInsulinSimulator(G0=initial_glucose)
    sim_pid.reset(G0=initial_glucose)
    pid_ctrl = PIDController(target=100, kp=OPTIMAL_KP, ki=OPTIMAL_KI, kd=OPTIMAL_KD, basal=OPTIMAL_BASAL, dt=5/60)
    pid_ctrl.reset()
    
    pid_glucose = []
    pid_insulin = []
    prev_glucose = None
    
    for j in range(len(kcal)):
        glucose = sim_pid.G
        pid_basal = pid_ctrl.compute(glucose, prev_glucose)
        bolus = kcal[j] * OPTIMAL_BOLUS_RATIO if kcal[j] > 0 else 0
        pid_total = pid_basal + bolus
        prev_glucose = glucose
        
        meal_glucose_impact = kcal[j] * 0.1 if kcal[j] > 0 else 0
        G, I = sim_pid.step(insulin_U=pid_total, meal_g=meal_glucose_impact)
        pid_glucose.append(G)
        pid_insulin.append(pid_total)
    
    # RL Controller
    sim_rl = ImprovedGlucoseInsulinSimulator(G0=initial_glucose)
    sim_rl.reset(G0=initial_glucose)
    rl_glucose = []
    rl_insulin = []
    rl_state = [sim_rl.G] * HIST_LEN
    
    for j in range(len(kcal)):
        glucose = sim_rl.G
        rl_input = np.array(rl_state + [kcal[j]], dtype=np.float32)
        action = trained_agent.select_action(rl_input)
        insulin = action_space[action]
        
        meal_glucose_impact = kcal[j] * 0.1 if kcal[j] > 0 else 0
        G, I = sim_rl.step(insulin_U=insulin, meal_g=meal_glucose_impact)
        rl_glucose.append(G)
        rl_insulin.append(insulin)
        rl_state.pop(0)
        rl_state.append(G)
    
    # Calculate and print metrics
    original_mean = np.mean(original_glucose)
    original_tir_total = np.mean([70 <= g <= 180 for g in original_glucose]) * 100
    pid_mean = np.mean(pid_glucose)
    pid_tir_total = np.mean([70 <= g <= 180 for g in pid_glucose]) * 100
    rl_mean = np.mean(rl_glucose)
    rl_tir_total = np.mean([70 <= g <= 180 for g in rl_glucose]) * 100
    
    print(f"  Original: Mean glucose = {original_mean:.1f}, TIR = {original_tir_total:.1f}%")
    print(f"  PID: Mean glucose = {pid_mean:.1f}, TIR = {pid_tir_total:.1f}%")
    print(f"  RL: Mean glucose = {rl_mean:.1f}, TIR = {rl_tir_total:.1f}%")


    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Time axis (convert to hours from start)
    time_hours = np.arange(len(timestamps)) * 5 / 60  # 5-minute intervals

    # Plot 1: Glucose traces comparison
    axes[0].plot(time_hours, original_glucose, 'gray', alpha=0.7, label='Original CGM', linewidth=1)
    axes[0].plot(time_hours, pid_glucose, 'blue', label='PID Controller', linewidth=1.5)
    axes[0].plot(time_hours, rl_glucose, 'red', label='RL Controller', linewidth=1.5)
    axes[0].axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Target (100)')
    axes[0].axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Lower bound (70)')
    axes[0].axhline(y=180, color='orange', linestyle='--', alpha=0.7, label='Upper bound (180)')
    axes[0].fill_between(time_hours, 70, 180, alpha=0.1, color='green', label='Target Range')
    axes[0].set_ylabel('Glucose (mg/dL)')
    axes[0].set_title(f'Glucose Traces Comparison - Subject {subject_id}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(50, 200)
    # Set y-axis upper bound to max CGM value for this subject
    ymax = np.nanmax(original_glucose)
    axes[0].set_ylim(bottom=0, top=ymax * 1.05)

    # Plot 2: Insulin delivery
    axes[1].plot(time_hours, pid_insulin, 'blue', label='PID Insulin', linewidth=1.5)
    axes[1].plot(time_hours, rl_insulin, 'red', label='RL Insulin', linewidth=1.5)
    axes[1].set_ylabel('Insulin (units)')
    axes[1].set_title('Insulin Delivery')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Food intake (kcal)
    axes[2].bar(time_hours, kcal, alpha=0.6, color='orange', label='Food Intake (kcal)')
    axes[2].set_ylabel('Calories (kcal)')
    axes[2].set_title('Food Intake')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)


    # Plot 4: Glucose improvement analysis
    # Calculate rolling time in range for each approach
    def rolling_tir(glucose_values, window=12):  # 1-hour rolling window
        tir_values = []
        for i in range(len(glucose_values)):
            start_idx = max(0, i - window + 1)
            window_glucose = glucose_values[start_idx:i + 1]
            tir = np.mean([70 <= g <= 180 for g in window_glucose]) * 100
            tir_values.append(tir)
        return tir_values


    original_tir = rolling_tir(original_glucose)
    pid_tir = rolling_tir(pid_glucose)
    rl_tir = rolling_tir(rl_glucose)

    # Add x-axis label
    axes[-1].set_xlabel("Time (hours)")

    plt.tight_layout()
    plt.savefig(f'aireadi/subject_{subject_id}_comprehensive_traces_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
# Create summary comparison plot
print(f"\nCreating summary comparison plot...")

# Calculate metrics for all subjects
all_metrics = []

for subject_id in subjects:
    sub = df[df['ID'] == subject_id].sort_values('timestamp').reset_index(drop=True)
    kcal = sub['kcal'].values
    original_glucose = sub['cgm'].values
    initial_glucose = sub['cgm'].iloc[0]


    # Run controllers (simplified for speed)
    # PID
    sim_pid = ImprovedGlucoseInsulinSimulator(G0=initial_glucose)
    sim_pid.reset(G0=initial_glucose)
    pid_ctrl = PIDController(target=100, kp=OPTIMAL_KP, ki=OPTIMAL_KI, kd=OPTIMAL_KD, basal=OPTIMAL_BASAL, dt=5/60)
    pid_ctrl.reset()
    
    pid_glucose = []
    prev_glucose = None
    
    for j in range(len(kcal)):
        glucose = sim_pid.G
        pid_basal = pid_ctrl.compute(glucose, prev_glucose)
        bolus = kcal[j] * OPTIMAL_BOLUS_RATIO if kcal[j] > 0 else 0
        pid_total = pid_basal + bolus
        prev_glucose = glucose
        
        meal_glucose_impact = kcal[j] * 0.1 if kcal[j] > 0 else 0
        G, I = sim_pid.step(insulin_U=pid_total, meal_g=meal_glucose_impact)
        pid_glucose.append(G)
    
    # RL
    sim_rl = ImprovedGlucoseInsulinSimulator(G0=initial_glucose)
    sim_rl.reset(G0=initial_glucose)
    rl_glucose = []
    rl_state = [sim_rl.G] * HIST_LEN
    
    for j in range(len(kcal)):
        glucose = sim_rl.G
        rl_input = np.array(rl_state + [kcal[j]], dtype=np.float32)
        action = trained_agent.select_action(rl_input)
        insulin = action_space[action]
        
        meal_glucose_impact = kcal[j] * 0.1 if kcal[j] > 0 else 0
        G, I = sim_rl.step(insulin_U=insulin, meal_g=meal_glucose_impact)
        rl_glucose.append(G)
        rl_state.pop(0)
        rl_state.append(G)
    
    # Calculate metrics
    original_mean = np.mean(original_glucose)
    original_tir = np.mean([70 <= g <= 180 for g in original_glucose]) * 100
    pid_mean = np.mean(pid_glucose)
    pid_tir = np.mean([70 <= g <= 180 for g in pid_glucose]) * 100
    rl_mean = np.mean(rl_glucose)
    rl_tir = np.mean([70 <= g <= 180 for g in rl_glucose]) * 100
    
    all_metrics.append({
        'subject': subject_id,
        'original_mean': original_mean,
        'original_tir': original_tir,
        'pid_mean': pid_mean,
        'pid_tir': pid_tir,
        'rl_mean': rl_mean,
        'rl_tir': rl_tir
    })

metrics_df = pd.DataFrame(all_metrics)

# Create summary comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Glucose improvement
axes[0,0].scatter(metrics_df['original_mean'], metrics_df['pid_mean'], alpha=0.6, label='PID', color='blue')
axes[0,0].scatter(metrics_df['original_mean'], metrics_df['rl_mean'], alpha=0.6, label='RL', color='red')
axes[0,0].plot([70, 200], [70, 200], 'k--', alpha=0.5, label='No change')
axes[0,0].axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Target (100)')
axes[0,0].set_xlabel('Original Mean Glucose (mg/dL)')
axes[0,0].set_ylabel('Controller Mean Glucose (mg/dL)')
axes[0,0].set_title('Glucose Improvement')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Time in range improvement
axes[0,1].scatter(metrics_df['original_tir'], metrics_df['pid_tir'], alpha=0.6, label='PID', color='blue')
axes[0,1].scatter(metrics_df['original_tir'], metrics_df['rl_tir'], alpha=0.6, label='RL', color='red')
axes[0,1].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='No change')
axes[0,1].axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Target TIR (70%)')
axes[0,1].set_xlabel('Original Time in Range (%)')
axes[0,1].set_ylabel('Controller Time in Range (%)')
axes[0,1].set_title('Time in Range Improvement')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Distribution of improvements
pid_improvement = metrics_df['pid_tir'] - metrics_df['original_tir']
rl_improvement = metrics_df['rl_tir'] - metrics_df['original_tir']

axes[1,0].hist(pid_improvement, bins=15, alpha=0.7, label='PID', color='blue')
axes[1,0].hist(rl_improvement, bins=15, alpha=0.7, label='RL', color='red')
axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.7, label='No improvement')
axes[1,0].set_xlabel('TIR Improvement (%)')
axes[1,0].set_ylabel('Number of Subjects')
axes[1,0].set_title('Distribution of Time in Range Improvements')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Glucose vs TIR trade-off
axes[1,1].scatter(metrics_df['pid_mean'], metrics_df['pid_tir'], alpha=0.6, label='PID', color='blue')
axes[1,1].scatter(metrics_df['rl_mean'], metrics_df['rl_tir'], alpha=0.6, label='RL', color='red')
axes[1,1].axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Target TIR (70%)')
axes[1,1].axvline(x=100, color='green', linestyle='--', alpha=0.7, label='Target Glucose (100)')
axes[1,1].set_xlabel('Mean Glucose (mg/dL)')
axes[1,1].set_ylabel('Time in Range (%)')
axes[1,1].set_title('Glucose vs Time in Range Trade-off')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aireadi/comprehensive_results_summary_v2.png', dpi=300, bbox_inches='tight')
plt.close()

# Save metrics
metrics_df.to_csv('aireadi/comprehensive_metrics_v2.csv', index=False)

print(f"\nVisualization complete!")
print(f"Individual subject traces saved to:")
for subject_id in top_subjects:
    print(f"  - aireadi/subject_{subject_id}_comprehensive_traces_v2.png")
print(f"Summary comparison saved to:")
print(f"  - aireadi/comprehensive_results_summary_v2.png")
print(f"Detailed metrics saved to:")
print(f"  - aireadi/comprehensive_metrics_v2.csv")

# Print summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Average TIR improvement:")
print(f"  PID: {pid_improvement.mean():.1f} ± {pid_improvement.std():.1f}%")
print(f"  RL: {rl_improvement.mean():.1f} ± {rl_improvement.std():.1f}%")
print(f"Subjects with >70% TIR:")
print(f"  PID: {(metrics_df['pid_tir'] > 70).sum()}/{len(metrics_df)} ({(metrics_df['pid_tir'] > 70).mean()*100:.1f}%)")
print(f"  RL: {(metrics_df['rl_tir'] > 70).sum()}/{len(metrics_df)} ({(metrics_df['rl_tir'] > 70).mean()*100:.1f}%)") 