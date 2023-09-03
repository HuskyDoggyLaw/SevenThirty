import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="rgb_array").env
env.reset()
rendered_image = env.render()
image_pil = Image.fromarray(rendered_image)
image_pil.show()

# Initialize the q-table with zero values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # learning-rate
gamma = 0.7  # discount-factor
epsilon = 0.1  # explor vs exploit

# Random generator
rng =np.random.default_rng()

# Perform 100,000 episodes
for i in range(100_000):
    # Reset the environment
    state = env.reset()

    done = False
    
    # Loop as long as the game is not over, i.e. done is not True
    while not done:
        if rng.random() < epsilon:
            action = env.action_space.sample() # Explore the action space
        else:
            # action = np.argmax(q_table[state]) # Exploit learned values
            action = env.action_space.sample() # Explore the action space

        # Apply the action and see what happens
        step_result = env.step(action) 
        next_state, reward, done, info =  step_result
        
        current_value = q_table[state, action]  # current Q-value for the state/action couple
        next_max = np.max(q_table[next_state])  # next best Q-value
        
        # Compute the new Q-value with the Bellman equation
        q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_max)



