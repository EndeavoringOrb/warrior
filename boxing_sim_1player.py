import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import rewards


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

    def forward(self, x):
        if x.device.type != device.type:
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
    target_q_vals = current_q_vals.detach().clone()
    actions_taken = torch.tensor(actions_taken, dtype=torch.uint8)
    
    with torch.no_grad():
        next_q_vals = q_network(next_state)
        maxes, _ = torch.max(next_q_vals.reshape(-1, action_options), dim=1)
        avg_max = maxes.mean()
        target_q_values = reward + discount_factor * avg_max - current_q_vals[torch.arange(0, len(q_vals), action_options) + actions_taken]
        target_q_vals[torch.arange(0, len(q_vals), action_options) + actions_taken] = current_q_vals[torch.arange(0, len(q_vals), action_options) + actions_taken] + learning_rate * target_q_values
    
    #print(target_q_vals-current_q_vals)

    return current_q_vals, target_q_vals

# Define hyperparameters and training settings
input_size = 55  # Size of input state
hidden_layers = [2048, 2048]  # Number of units in the hidden layer
output_size = 63  # Number of outputs
learning_rate = 1e-6
discount_factor = 0.95
num_episodes = 1_000_000
episode_length = 1000
epsilon = 0.1
display_simulation = True
delay_for_cpu_to_live_laugh_love = 0.01 if display_simulation else 0.01# seconds

load_model = True if input("Load model? [Y/n]: ").lower() == 'y' else False
if load_model:
    network1 = torch.load('models/torchmodel.pt')
else:
    # Initialize the networks and optimizer
    network1 = BoxerNetwork(input_size, hidden_layers, output_size)
network1 = network1.to(device)
optimizer1 = optim.Adam(network1.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Create the Mujoco model and data
m = mujoco.MjModel.from_xml_path('humanoid.xml')
d = mujoco.MjData(m)

# Training loop
for episode in range(num_episodes):
    episode_start = time.time()
    # reset environment
    d = mujoco.MjData(m)
    if display_simulation:
        if episode > 0:
            viewer.close()
        viewer = mujoco.viewer.launch_passive(m, d)

    state = np.concatenate((d.qpos, d.qvel), axis=0, dtype=np.float64)
    state = torch.from_numpy(state).float()
    state1 = state
    total_reward1 = 0
    reward1 = 0
    standing_last = d.xpos[2][2]
    mean_diff = 0

    for step in range(episode_length):
        print(f"{step+1}/{episode_length} - avg reward: {total_reward1/(step+1):.5f}",end="\r")
        time.sleep(delay_for_cpu_to_live_laugh_love)
        # Choose actions using the neural networks
        state1.to(device)
        q_vals_tensor = network1(state1)
        q_vals = torch.Tensor.cpu(q_vals_tensor).detach().numpy()

        step_start = time.time()

        # Apply actions to the Mujoco model
        ep_val = np.random.random()
        choice = np.random.choice(3)
        maxes = [[(np.argmax(q_vals[i:i+3]) if ep_val > epsilon else choice), (max(q_vals[i:i+3]) if ep_val > epsilon else q_vals[i+choice])] for i in range(0, len(q_vals), 3)]
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

        reward1 = rewards.scale_number(d.xpos[2][2], _STAND_HEIGHT, -_STAND_HEIGHT/4)
        upright_reward = (d.xpos[2][2]-d.xpos[1][2]) * (1/0.19)
        reward1 = (reward1*0.5 + upright_reward*0.5) / 2

        # Update total rewards
        #total_reward1 += upright_reward
        total_reward1 += reward1
        

        # Convert new state to torch tensor
        state = np.concatenate((d.qpos, d.qvel), axis=0, dtype=np.float64)
        new_state1 = torch.from_numpy(state).float()

        current1, target1 = update_q_net_mult_actions(network1, state1, 3, [i[0] for i in maxes], q_vals_tensor, reward1, new_state1, learning_rate, discount_factor)

        loss1 = criterion(current1, target1)

        # Update neural network weights
        optimizer1.zero_grad()
        loss1.backward()
        a = optimizer1.step()

        # Update state
        state1 = new_state1

        if display_simulation:
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    episode_end = time.time()
    print(f"\nEpisode: {episode+1}, Reward1: {total_reward1}, Time Taken: {episode_end-episode_start}")
    print("saving model...")
    torch.save(network1, f"models/torchmodel.pt")#_ep{episode+1}