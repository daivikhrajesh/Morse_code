from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from stable_baselines3 import DQN
import gymnasium as gym

# Load models
classification_model = tf.keras.models.load_model('../models/morse_model.h5')
rl_model = DQN.load('../models/morse_rl_model')

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')

class MorseEnv(gym.Env):
    def __init__(self):
        super(MorseEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # Dot or Dash
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(10,))
        self.current_step = 0
        self.max_steps = 10
        self.signal = np.zeros(self.max_steps)
        self.expected_letter = 'A'

    def step(self, action):
        self.signal[self.current_step] = action
        self.current_step += 1
        done = self.current_step == self.max_steps
        reward = self._get_reward() if done else 0
        return self.signal, reward, done, {}

    def reset(self):
        self.signal = np.zeros(self.max_steps)
        self.current_step = 0
        return self.signal

    def _get_reward(self):
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences([self.signal], maxlen=10)
        input_sequence = np.expand_dims(input_sequence, -1)
        prediction = classification_model.predict(input_sequence)
        letter = chr(np.argmax(prediction) + ord('A'))
        return 1 if letter == self.expected_letter else -1

env = MorseEnv()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interpret', methods=['POST'])
def interpret():
    data = request.get_json()
    signal = np.array(data['signal'])

    # Use the RL model for interpretation
    obs = env.reset()
    for s in signal:
        obs, reward, done, info = env.step(s)
        if done:
            break
    
    # Get the predicted letter
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences([env.signal], maxlen=10)
    input_sequence = np.expand_dims(input_sequence, -1)
    prediction = classification_model.predict(input_sequence)
    letter = chr(np.argmax(prediction) + ord('A'))
    
    return jsonify({'text': letter})

if __name__ == '__main__':
    app.run(debug=True)
