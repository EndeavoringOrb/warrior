from stable_baselines3 import PPO
from SB3_env import CustomEnv

# Create the custom environment
env = CustomEnv()

# Create and train the agent using the PPO algorithm
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Use the trained agent to interact with the environment
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

# Close the environment
env.close()