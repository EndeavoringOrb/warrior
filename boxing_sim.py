import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#output space 21 actuators
"""
body_pos
body_inertia
actuator_gear
actuator_acc0
actuator_ctrlrange
"""
# Define the neural network architecture
class BoxerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BoxerNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def my_update_q_network(q_network, state, action, reward, next_state, learning_rate, discount_factor):
    # Convert state, action, reward, and next_state to PyTorch tensors
    state = torch.tensor(state, dtype=torch.float32)
    reward = torch.tensor(reward, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)

    # Calculate current and updated Q-values
    current_q_vals = q_network(state)
    target_q_vals = current_q_vals.clone()
    with torch.no_grad():
        next_q_vals = q_network(next_state)
        target_q_value = reward + discount_factor * torch.max(next_q_vals) - current_q_vals[action[0],action[1]]
        target_q_vals.index_put_((action[0],action[1]), current_q_vals[action[0],action[1]] + learning_rate * target_q_value)
    
    return current_q_vals, target_q_vals

# Define hyperparameters and training settings
input_size = 4  # Size of input state
hidden_size = 32  # Number of units in the hidden layer
output_size = 2  # Number of actions (boxing moves)
learning_rate = 0.001
discount_factor = 0.95
num_episodes = 1000
episode_length = 1000

# Initialize the networks and optimizer
network1 = BoxerNetwork(input_size, hidden_size, output_size)
network2 = BoxerNetwork(input_size, hidden_size, output_size)
optimizer1 = optim.Adam(network1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(network2.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Create the Mujoco model and data
m = mujoco.MjModel.from_xml_path('humanoid.xml')
d = mujoco.MjData(m)

# Training loop
for episode in range(num_episodes):
    state = 
    state1 = torch.from_numpy(d.qpos[:input_size]).float()
    state2 = torch.from_numpy(d.qpos[:input_size]).float()
    total_reward1 = 0
    total_reward2 = 0

    for step in range(episode_length):
        # Choose actions using the neural networks
        action1 = network1(state1).detach().numpy()
        action2 = network2(state2).detach().numpy()

        # Apply actions to the Mujoco model
        d.ctrl[:output_size] = action1
        d.ctrl[output_size:2*output_size] = action2
        mujoco.mj_step(m, d)

        # Get rewards based on collision detection
        reward1 = -1 if d.ncon > 0 else 0
        reward2 = -1 if d.ncon > 0 else 0

        # Update total rewards
        total_reward1 += reward1
        total_reward2 += reward2

        # Convert new state to torch tensor
        new_state1 = torch.from_numpy(d.qpos[:input_size]).float()
        new_state2 = torch.from_numpy(d.qpos[:input_size]).float()

        # Update neural network weights
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        current1, target1 = my_update_q_network(network1, state1, action1, reward1, new_state1, learning_rate, discount_factor)
        current2, target2 = my_update_q_network(network2, state2, action2, reward2, new_state2, learning_rate, discount_factor)

        loss1 = criterion(current1, target1)
        loss2 = criterion(current2, target2)

        loss1.backward()
        loss2.backward()

        optimizer1.step()
        optimizer2.step()

        # Update state
        state1 = new_state1
        state2 = new_state2

    print(f"Episode: {episode+1}, Reward1: {total_reward1}, Reward2: {total_reward2}")

# Test the trained networks
state1 = torch.from_numpy(d.qpos[:input_size]).float()
state2 = torch.from_numpy(d.qpos[:input_size]).float()
total_reward1 = 0
total_reward2 = 0

for step in range(episode_length):
    action1 = network1(state1).detach().numpy()
    action2 = network2(state2).detach().numpy()

    d.ctrl[:output_size] = action1
    d.ctrl[output_size:2*output_size] = action2
    mujoco.mj_step(m, d)

    reward1 = -1 if d.ncon > 0 else 0
    reward2 = -1 if d.ncon > 0 else 0

    total_reward1 += reward1
    total_reward2 += reward2

    new_state1 = torch.from_numpy(d.qpos[:input_size]).float()
    new_state2 = torch.from_numpy(d.qpos[:input_size]).float()

    state1 = new_state1
    state2 = new_state2

print(f"Test Result - Reward1: {total_reward1}, Reward2: {total_reward2}")