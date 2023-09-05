#------------------------------------------------------------------------------------------------------------------
#   Taxi example for reinforcement learning.
#------------------------------------------------------------------------------------------------------------------
import gym
import os
from time import sleep

import colorama
colorama.init()

# Helper function 
def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)

# Create environment
env = gym.make("Taxi-v3").env

# Set inicial state
env.s = env.encode(3, 1, 2, 0)  
env.render()

# Find solution using random walk
epochs = 0
penalties = 0
reward = 0

done = False

while not done:    

    # Select one random action
    action = env.action_space.sample()

    # Perform selected action
    state, reward, done, info = env.step(action)

    # Update number of epochs and penalties
    if reward == -10:
        penalties += 1
    
    epochs += 1

    # Show current state
    clearConsole()
    env.render()       
    print('State: {}  Action: {}  Reward: {}'.format(state, action, reward))
    sleep(.02)    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

input()
#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------