import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from improved_simulator import ImprovedGlucoseInsulinSimulator
from dqn_agent import DQNAgent, ReplayBuffer
import torch

# Parameters
HIST_LEN = 5
EPISODES = 1000
PLOT_DIR = 'aireadi'
os.makedirs(PLOT_DIR, exist_ok=True)

# Load rolling window data
df = pd.read_csv('aireadi_cgm_kcal.csv', parse_dates=['timestamp'])
subjects = df['ID'].unique()

# Create simplified environment that uses kcal directly
class SimplifiedRollingWindowGlucoseEnv:
    def __init__(self, history_len=5):
        self.history_len = history_len
        # Realistic insulin action space
        self.action_space = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.n_actions = len(self.action_space)
        self.simulator = ImprovedGlucoseInsulinSimulator()
        
    def reset(self, initial_glucose=None):
        if initial_glucose is None:
            initial_glucose = np.random.uniform(80, 200)
        self.simulator.reset(G0=initial_glucose)
        self.glucose_history = [initial_glucose] * self.history_len
        self.kcal = 0.0
        return self._get_state()
    
    def step(self, action, kcal):
        # Get insulin dose
        insulin = self.action_space[action]
        
        # Convert kcal directly to glucose impact (simplified)
        # Assume 1 kcal raises glucose by ~0.1 mg/dL (realistic approximation)
        meal_glucose_impact = kcal * 0.1 if kcal > 0 else 0
        
        # Simulate one step with direct kcal conversion
        glucose, insulin_level = self.simulator.step(insulin_U=insulin, meal_g=meal_glucose_impact)
        
        # Update history
        self.glucose_history.pop(0)
        self.glucose_history.append(glucose)
        self.kcal = kcal
        
        # Calculate reward
        reward = self._calculate_reward(glucose)
        
        # Check if episode should end
        done = glucose < 50 or glucose > 300
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        # State: [glucose_history, kcal]
        return np.array(self.glucose_history + [self.kcal], dtype=np.float32)
    
    def _calculate_reward(self, glucose):
        # Improved reward function for simplified approach
        target = 100
        
        # Base reward based on distance from target
        distance = abs(glucose - target)
        base_reward = -distance / 100  # Normalize by 100
        
        # Bonus for being in target range (70-100)
        if 70 <= glucose <= 180:
            base_reward += 8.0  # Increased bonus for good control
        
        # Penalty for hypoglycemia
        if glucose < 70:
            base_reward -= 15.0  # Increased penalty for safety
        
        # Penalty for severe hyperglycemia
        if glucose > 240:
            base_reward -= 8.0
        
        # Small penalty for insulin usage (encourage efficiency)
        insulin_penalty = -0.02  # Very small penalty to encourage insulin use
        
        return base_reward + insulin_penalty

# Create environment and agent
env = SimplifiedRollingWindowGlucoseEnv(history_len=HIST_LEN)
state_dim = HIST_LEN + 1  # 5 glucose values + 1 kcal
action_dim = env.n_actions

agent = DQNAgent(state_dim, action_dim)
agent.epsilon = 0.1  # Start with some exploration
replay_buffer = ReplayBuffer(capacity=10000)

# Training loop
training_rewards = []
episode_lengths = []

print("Training simplified RL agent with direct kcal approach...")
print(f"Episodes: {EPISODES}")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Action space: {env.action_space}")

for episode in range(EPISODES):
    # Select random subject and time window
    subject = np.random.choice(subjects)
    subject_data = df[df['ID'] == subject].sort_values('timestamp').reset_index(drop=True)
    
    if len(subject_data) < 50:  # Skip subjects with too little data
        continue
    
    # Start from random position
    start_idx = np.random.randint(0, len(subject_data) - 50)
    episode_data = subject_data.iloc[start_idx:start_idx + 50]
    
    # Reset environment with initial glucose
    initial_glucose = episode_data['cgm'].iloc[0]
    state = env.reset(initial_glucose=initial_glucose)
    
    episode_reward = 0
    episode_length = 0
    
    for i, (_, row) in enumerate(episode_data.iterrows()):
        # Select action
        action = agent.select_action(state)
        
        # Take step
        next_state, reward, done = env.step(action, row['kcal'])
        
        # Store experience
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Train agent
        if len(replay_buffer) > 64:  # batch_size
            agent.update(replay_buffer, batch_size=64)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        if done:
            break
    
    training_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    # Decay epsilon
    if episode % 100 == 0:
        agent.epsilon = max(0.01, agent.epsilon * 0.95)
    
    if episode % 100 == 0:
        avg_reward = np.mean(training_rewards[-100:])
        avg_length = np.mean(episode_lengths[-100:])
        print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}, Epsilon = {agent.epsilon:.3f}")

# Save trained agent
torch.save(agent.q_net.state_dict(), os.path.join(PLOT_DIR, 'rl_agent_simplified_v2.pth'))
print(f"Trained simplified agent saved to {os.path.join(PLOT_DIR, 'rl_agent_simplified_v2.pth')}")

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(training_rewards)
plt.title('Simplified RL Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.title('Simplified RL Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rl_training_progress_simplified.png'))
plt.close()

print("Simplified RL training completed!")
print(f"Final average reward: {np.mean(training_rewards[-100:]):.2f}")
print(f"Final average episode length: {np.mean(episode_lengths[-100:]):.1f}")

# Test the trained agent on a few subjects
print("\nTesting trained agent on sample subjects...")
test_subjects = subjects[:5]

for sid in test_subjects:
    sub = df[df['ID'] == sid].sort_values('timestamp').reset_index(drop=True)
    kcal = sub['kcal'].values
    initial_glucose = sub['cgm'].iloc[0]
    
    # Test trained agent
    sim = ImprovedGlucoseInsulinSimulator(G0=initial_glucose)
    sim.reset(G0=initial_glucose)
    rl_glucose = []
    rl_insulin = []
    rl_state = [sim.G] * HIST_LEN
    
    for i in range(min(50, len(kcal))):  # Test first 50 steps
        glucose = sim.G
        rl_input = np.array(rl_state + [kcal[i]], dtype=np.float32)
        action = agent.select_action(rl_input)
        insulin = env.action_space[action]
        
        meal_glucose_impact = kcal[i] * 0.1 if kcal[i] > 0 else 0
        G, I = sim.step(insulin_U=insulin, meal_g=meal_glucose_impact)
        rl_glucose.append(G)
        rl_insulin.append(insulin)
        rl_state.pop(0)
        rl_state.append(G)
    
    mean_glucose = np.mean(rl_glucose)
    tir = np.mean([70 <= g <= 180 for g in rl_glucose]) * 100
    mean_insulin = np.mean(rl_insulin)
    
    print(f"Subject {sid}: Mean glucose = {mean_glucose:.1f}, Time in range = {tir:.1f}%, Mean insulin = {mean_insulin:.3f}") 