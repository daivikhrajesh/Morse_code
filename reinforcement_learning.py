import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('../models/morse_model.h5')

class MorseEnv(gym.Env):
    def __init__(self):
        super(MorseEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Dot, Dash, or Space
        self.observation_space = spaces.Box(low=0, high=5, shape=(10,), dtype=np.float32)
        self.current_step = 0
        self.max_steps = 10
        self.signal = np.zeros(self.max_steps, dtype=np.float32)
        self.expected_letter = 'A'  # Change this to dynamically set during training

    def step(self, action):
        # Map action to signal values
        if action == 0:
            action_value = 1  # Dot
        elif action == 1:
            action_value = 2  # Dash
        else:
            action_value = 5  # Space

        # Store action value in the signal array
        self.signal[self.current_step] = action_value
        self.current_step += 1
        done = self.current_step == self.max_steps
        
        reward = self._get_reward() if done else 0
        return self.signal, reward, done, {}

    def reset(self):
        self.signal = np.zeros(self.max_steps, dtype=np.float32)
        self.current_step = 0
        return self.signal

    def _get_reward(self):
        # Simulate reward based on model prediction
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences([self.signal], maxlen=10, padding='post')
        input_sequence = np.expand_dims(input_sequence, -1)  # Add channel dimension
        prediction = model.predict(input_sequence)
        letter = chr(np.argmax(prediction) + ord('A'))
        return 1 if letter == self.expected_letter else -1

# Create environment
env = MorseEnv()

# Initialize RL model
model_rl = DQN('MlpPolicy', env, verbose=1)

# Train model
model_rl.learn(total_timesteps=10000)

# Save the trained RL model
model_rl.save('../models/morse_rl_model')
