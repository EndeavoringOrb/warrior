import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite
import numpy as np

# Define the AI model architecture
class AIModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AIModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.fc2 = nn.Linear(1, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_state(time_step):
    # Convert observation to a PyTorch tensor
    state = []
    for k, v in time_step.observation.items():
        if v.shape == ():
            state.append(v)
        else:
            for i in v:
                state.append(i)
    state = torch.FloatTensor(state)
    return state

# Set up the environment
env = suite.load(domain_name="humanoid", task_name="stand")

# Extract observation and action space dimensions
obs_spec = env.observation_spec()
action_spec = env.action_spec()
state_size = 21+1+12+3+3+27
action_size = 21

# Hyperparameters
learning_rate = 0.001
num_episodes = 1000
max_steps = 1000
gamma = 0.99

# Initialize the AI model
model = AIModel(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

# Training loop
for episode in range(num_episodes):
    time_step = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Convert observation to a PyTorch tensor
        state = get_state(time_step)

        # Forward pass through the AI model to get the action
        action_tensor = model(state)
        action = action_tensor.detach().numpy()[0]

        # Take the action in the environment
        time_step = env.step(action)

        # Extract next state, reward, and done flag=
        next_state = get_state(time_step)

        reward = time_step.reward
        done = time_step.last()

        # Convert next state to a PyTorch tensor
        next_state_tensor = torch.FloatTensor(next_state.reshape(1, -1))

        # Calculate the TD target
        target = reward + gamma * model(next_state_tensor).max()

        # Calculate the loss
        loss = mse_loss(action_tensor, target)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the reward
        episode_reward += reward

        if done:
            break

    # Print episode information
    print(f"Episode: {episode+1}, Reward: {episode_reward}")

# Test the trained model
time_step = env.reset()
total_reward = 0

for step in range(max_steps):
    # Convert observation to a PyTorch tensor
    observation_tensor = torch.FloatTensor(time_step.observation['observations'].reshape(1, -1))

    # Forward pass through the AI model to get the action
    action_tensor = model(observation_tensor)
    action = action_tensor.detach().numpy()[0]

    # Take the action in the environment
    time_step = env.step(action)

    # Update the reward
    total_reward += time_step.reward

    if time_step.last():
        break

# Print the total reward of the test run
print(f"Total Test Reward: {total_reward}")

# Close the environment
env.close()