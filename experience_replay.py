import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from dm_control.utils import rewards

#output space 21 actuators
"""
body_pos
body_inertia
actuator_gear
actuator_acc0
actuator_ctrlrange
"""

_STAND_HEIGHT = 1.4
force_cpu = False
if force_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")

# Define the neural network architecture
class BoxerNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(BoxerNetwork, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module(f'linear0', nn.Linear(input_size, hidden_layers[0]))
        for i in range(len(hidden_layers)-1):
            self.model.add_module(f'linear{i+1}', nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.model.add_module(f'linear{len(hidden_layers)}', nn.Linear(hidden_layers[-1], output_size))
        self.model.add_module(f"sig0", nn.Sigmoid())
        self.model.to(device)

    def forward(self, x):
        if x.device != device:
            x = x.to(device)
        x = self.model(x)
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
        target_q_value = reward + discount_factor * torch.max(next_q_vals) - current_q_vals[action]
        target_q_vals[action] = current_q_vals[action] + learning_rate * target_q_value
    
    return current_q_vals, target_q_vals

def update_q_net_mult_actions(q_network, state, action_options, actions_taken, q_vals, reward, next_state, learning_rate, discount_factor):
    """
    action_options is how many action options there are per robot input
    actions is a list of chosen actions. each item is the index of which action option was chosen
    """
    reward = torch.tensor(reward, dtype=torch.float32)
    #reward = reward.to(device)
    next_state = next_state.to(device)
    current_q_vals = q_vals
    #current_q_vals = current_q_vals.to(device)
    #target_q_vals = current_q_vals.clone().to(device)
    target_q_vals = current_q_vals.clone()
    actions_taken = torch.tensor(actions_taken, dtype=torch.uint8)
    
    with torch.no_grad():
        next_q_vals = q_network(next_state)
        maxes, _ = torch.max(next_q_vals.reshape(-1, action_options), dim=1)
        avg_max = maxes.mean()
        target_q_values = reward + discount_factor * avg_max - current_q_vals[torch.arange(0, len(q_vals), action_options) + actions_taken]
        target_q_vals[torch.arange(0, len(q_vals), action_options) + actions_taken] = current_q_vals[torch.arange(0, len(q_vals), action_options) + actions_taken] + learning_rate * target_q_values
    
    
    return current_q_vals, target_q_vals

def train_on_batch(q_network, optimizer, criterion, states, q_vals, rewards, next_states, learning_rate, discount_factor, actions_taken, action_options):
    
    # Update neural network weights
    optimizer.zero_grad()

    batch_size = len(rewards)

    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.stack(next_states, dim=0)
    next_states = next_states.to(device)
    current_q_vals = torch.stack(q_vals, dim=0).to(device)
    target_q_vals = current_q_vals.clone().detach()
    actions_taken = torch.tensor(actions_taken, dtype=torch.uint8)

    next_q_vals = q_network(next_states)

    batch_maxes = []
    for i in range(batch_size):
        maxes, _ = torch.max(next_q_vals[i].reshape(-1, action_options), dim=1)
        batch_maxes.append(maxes.mean())
    target_q_values = []
    for i in range(batch_size):
        target_q_values.append(
            rewards[i] + discount_factor * batch_maxes[i].item() - current_q_vals[i][torch.arange(0, len(q_vals[i]), action_options) + actions_taken[i]]
        )
    target_q_values = torch.stack(target_q_values, dim=0)
    for i in range(len(target_q_values)):
        target_q_vals[i][torch.arange(0, len(q_vals[i]), action_options) + actions_taken[i]] = current_q_vals[i][torch.arange(0, len(q_vals[i]), action_options) + actions_taken[i]] + learning_rate * target_q_values[i]

    current_q_vals.requires_grad_()
    target_q_vals.requires_grad_()

    loss = criterion(current_q_vals, target_q_vals)

    loss.backward()

    optimizer1.step()

# Define hyperparameters and training settings
input_size = 55  # Size of input state
hidden_layers = [32, 32]  # Number of units in the hidden layer
output_size = 63  # Number of outputs
learning_rate = 0.001
discount_factor = 0.95
num_episodes = 1_000_000
episode_length = 1000
display_simulation = False

# Initialize the networks and optimizer
network1 = BoxerNetwork(input_size, hidden_layers, output_size)
#network2 = BoxerNetwork(input_size, hidden_layers, output_size)
network1 = network1.to(device)
#network2 = network2.to(device)
optimizer1 = optim.Adam(network1.parameters(), lr=learning_rate)
#optimizer2 = optim.Adam(network2.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
criterion.to(device)

# Create the Mujoco model and data
m = mujoco.MjModel.from_xml_path('humanoid.xml')
d = mujoco.MjData(m)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # Training loop
    for episode in range(num_episodes):
        episode_start = time.time()
        # reset environment
        d = mujoco.MjData(m)
        if display_simulation:
            if episode > 0:
                viewer.close()
            viewer = mujoco.viewer.launch_passive(m, d)

        states = []
        actions = []
        rewards_lst = []
        next_states = []
        actions_taken = []

        state = np.concatenate((d.qpos, d.qvel), axis=0, dtype=np.float64)
        states.append(state)

        state = torch.from_numpy(state).float()
        total_reward1 = 0

        for step in range(episode_length):
            # Choose actions using the neural networks
            state.to(device)
            #network1.to(device)
            q_vals_tensor = network1(state)
            q_vals = torch.Tensor.cpu(q_vals_tensor).detach()
            actions.append(q_vals)

            step_start = time.time()

            # Apply actions to the Mujoco model
            maxes = [[np.argmax(q_vals[i:i+3]), max(q_vals[i:i+3])] for i in range(0, len(q_vals), 3)]
            actions_taken.append([i[0] for i in maxes])
            controls = []
            for i, q in maxes:
                scaled_q = (q + 1)/2
                if i == 0:
                    controls.append(-scaled_q)
                elif i == 1:
                    controls.append(scaled_q)
                else:
                    controls.append(0)
            d.ctrl = controls

            # Take a step
            mujoco.mj_step(m, d)
            
            # Get rewards based on collision detection
            standing = rewards.tolerance(d.xpos[2][2],
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=_STAND_HEIGHT/4)
            rewards_lst.append(standing)

            # Convert new state to torch tensor
            state = np.concatenate((d.qpos, d.qvel), axis=0, dtype=np.float64)
            new_state = torch.from_numpy(state).float()
            next_states.append(new_state)
            states.append(new_state)
            
            # Update state
            state = new_state

            if display_simulation:
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        episode_end = time.time()
        train_on_batch(network1, optimizer1, criterion, states, actions, rewards_lst, next_states, learning_rate, discount_factor, actions_taken, 3)
        print(f"Episode: {episode+1}, Reward1: {total_reward1}, Time Taken: {episode_end-episode_start}")

values = prof.key_averages().table(sort_by="cpu_time_total")
with open("profiler.txt", "w", encoding="utf-8") as f:
    f.writelines(values)
print(values)