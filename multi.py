import torch.multiprocessing as mp
from collections import deque
from random import sample
import torch
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch.nn as nn
import torch.optim as optim


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
        #self.model.add_module(f"sig0", nn.Sigmoid())
        #self.model.to(device)

    def forward(self, x):
        if x.device.type != device.type:
            x = x.to(device)
        x = self.model(x)
        return x


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
learning_rate = 0.001
discount_factor = 0.95
num_episodes = 1_000_000
episode_length = 1000
batch_size = 10
replay_memory_capacity = 100
display_simulation = True
delay_for_cpu_to_live_laugh_love = 0.01 if display_simulation else 0.01# seconds

# Initialize the networks and optimizer
network1 = BoxerNetwork(input_size, hidden_layers, output_size)
#network2 = BoxerNetwork(input_size, hidden_layers, output_size)
network1 = network1.to(device)
#network2 = network2.to(device)
optimizer1 = optim.Adam(network1.parameters(), lr=learning_rate)
#optimizer2 = optim.Adam(network2.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Create the Mujoco model and data
m = mujoco.MjModel.from_xml_path('humanoid.xml')
d = mujoco.MjData(m)

def worker(step, episode_length, delay_for_cpu_to_live_laugh_love, device, network1, m, d, rewards, standing_last, standing, episode_reward, replay_memory):
    print(f"{step+1}/{episode_length} - avg reward: {episode_reward/(step+1):5}", end="\r")
    time.sleep(delay_for_cpu_to_live_laugh_love)
    
    # Choose actions using the neural networks
    state1 = d.get_state().copy()
    state1 = torch.from_numpy(state1).float().to(device)
    q_vals_tensor = network1(state1)
    q_vals = q_vals_tensor.cpu().detach().numpy()

    # Apply actions to the Mujoco model
    maxes = [[np.argmax(q_vals[i:i+3]), max(q_vals[i:i+3])] for i in range(0, len(q_vals), 3)]
    controls = []
    for i, q in maxes:
        scaled_q = (q + 1) / 2
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
    standing_new = d.xpos[2][2]
    reward1 = standing_new - standing
    standing_last = standing_new
    reward1 = rewards.scale_number(reward1, 0.001, -0.001)

    # Update total rewards
    episode_reward += reward1

    # Convert new state to torch tensor
    new_state1 = torch.from_numpy(d.get_state().copy()).float().to(device)

    # Store the experience in replay memory
    replay_memory.append((state1, maxes, reward1, new_state1))

    # Update state
    d.set_state(new_state1.cpu().numpy())

def optimize_network(optimizer1, loss1_queue, replay_memory):
    while True:
        if len(replay_memory) >= batch_size:
            # Sample a batch from replay memory
            batch = sample(replay_memory, batch_size)
            states, actions, rewards, next_states = zip(*batch)

            # Prepare tensors
            states = torch.stack(states).to(device)
            actions = [item[0] for item in actions]
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            next_states = torch.stack(next_states).to(device)

            # Compute Q-values for the current and next states
            q_vals = network1(states)
            q_vals_next = network1(next_states)

            # Compute target Q-values
            target_q_vals = q_vals.clone()
            for i, action in enumerate(actions):
                for j, q in enumerate(action):
                    target_q_vals[i, j*3:j*3+3] = rewards[i] + discount_factor * torch.max(q_vals_next[i, j*3:j*3+3])

            # Compute loss and update network weights
            loss1 = criterion(q_vals, target_q_vals)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            # Enqueue the loss for logging
            loss1_queue.put(loss1.item())

def run_episodes(num_episodes, episode_length, delay_for_cpu_to_live_love_laugh, display_simulation):
    loss1_queue = mp.Queue()
    processes = []
    replay_memory = deque(maxlen=replay_memory_capacity)
    network1.share_memory()

    for episode in range(num_episodes):
        episode_start = time.time()
        d = mujoco.MjData(m)

        if display_simulation:
            if episode > 0:
                viewer.close()
            viewer = mujoco.viewer.launch_passive(m, d)

        total_reward1 = 0
        standing_last = d.xpos[2][2]

        for step in range(episode_length):
            p = mp.Process(target=worker, args=(step, episode_length, delay_for_cpu_to_live_love_laugh, device, network1, m, d, rewards, standing_last, d.xpos[2][2], total_reward1, replay_memory))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        optimize_network(optimizer1, loss1_queue, replay_memory)

        episode_end = time.time()
        print(f"\nEpisode: {episode+1}, Reward1: {total_reward1}, Time Taken: {episode_end-episode_start}")
        print("saving model...")
        torch.save(network1, "models/torchmodel_multi.pt")

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Initialize and configure your network, optimizer, and other variables here

    run_episodes(num_episodes, episode_length, delay_for_cpu_to_live_laugh_love, display_simulation)