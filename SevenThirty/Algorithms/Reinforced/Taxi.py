import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="rgb_array").env

env.reset()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

state = env.encode(1, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)
env.s = state


rendered_image = env.render()
image_pil = Image.fromarray(rendered_image)
image_pil.show()

# plt.imshow(rendered_image)
# plt.axis('off')  # Turn off axis labels
# plt.show()

env.close()


