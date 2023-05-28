import torch
from SB3_env import CustomEnv
from boxing_sim_1player import BoxerNetwork
from torch import optim, nn

# Define variables
episodes = 10
steps_per_episode = 1000
learning_rate = 1e-6

# Create the custom environment
env = CustomEnv()

# Create the model
load_model = True if input("Load model? [Y/n]: ").lower() == 'y' else False
if load_model:
    model = torch.load('models/torchmodel.pt')
else:
    # Initialize the networks and optimizer
    model = BoxerNetwork(55, [2048, 2048], 21)
optimizer1 = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Use the trained agent to interact with the environment
for ep in range(episodes):
    obs = env.reset()
    for _ in range(steps_per_episode):
        q_vals = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(q_vals)
        if done:
            obs = env.reset()

# Close the environment
env.close()
