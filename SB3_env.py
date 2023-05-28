import gym
from gym import spaces
import numpy as np
import mujoco
import rewards

class CustomEnv(gym.Env):
    def __init__(self, epsilon=0.1):
        super(CustomEnv, self).__init__()

        # Define the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32)

        # Define the action space as continuous
        self.action_space = spaces.Box(low=-1, high=1, shape=(21,), dtype=np.float32)
        # Create the Mujoco model and data
        self.m = mujoco.MjModel.from_xml_path('humanoid.xml')
        self.d = mujoco.MjData(self.m)
        self.epsilon = epsilon
        self.stand_height = 1.4

    def reset(self):
        # reset environment
        self.d = mujoco.MjData(self.m)
        state = np.concatenate((self.d.qpos, self.d.qvel), axis=0, dtype=np.float64)
        return state

    def step(self, q_vals):
        # Apply actions to the Mujoco model
        ep_val = np.random.random()
        choice = np.random.choice(3)
        maxes = [[(np.argmax(q_vals[i:i+3]) if ep_val > self.epsilon else choice), (max(q_vals[i:i+3]) if ep_val > self.epsilon else q_vals[i+choice])] for i in range(0, len(q_vals), 3)]
        controls = []
        for i, q in maxes:
            scaled_q = (q + 1)/2
            if i == 0:
                controls.append(-scaled_q)
            elif i == 1:
                controls.append(scaled_q)
            else:
                controls.append(0)
        self.d.ctrl = controls

        # Take a step
        mujoco.mj_step(self.m, self.d)

        state = np.concatenate((self.d.qpos, self.d.qvel), axis=0, dtype=np.float64)

        reward = rewards.scale_number(self.d.xpos[2][2], self.stand_height, -self.stand_height/4)
        upright_reward = (self.d.xpos[2][2]-self.d.xpos[1][2]) * (1/0.19)
        reward = (reward*0.5 + upright_reward*0.5) / 2

        done = False

        info = {}
        return state, reward, done, info

    def render(self, mode='human'):
        # Render the environment (optional)
        pass

    def close(self):
        # Clean up resources or close the environment (optional)
        pass
