#------------------------------------------------------------------------------------------------------------------
#   Taxi example for reinforcement learning.
#------------------------------------------------------------------------------------------------------------------
import gym
import numpy as np
import os
import random
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

###### Q-Learning ######

# Create Q-learning table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1 

# Training phase
for i in range(100000):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:

        # Select next action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Use action with the best Q-value

        # Perform selected action
        next_state, reward, done, info = env.step(action) 
        
        # Get old Q-Value
        old_value = q_table[state, action]

        # Update Q-Value
        next_max = np.max(q_table[next_state])        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # Update state, penalties and epochs
        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    # Show current training status
    if i % 100 == 0:
        print("Episode: {}".format(i))

print("Training finished.\n")

# Run environment using Q-table
state = env.encode(3, 1, 2, 0)  
env.s = state
env.render()

epochs = 0
penalties = 0
reward = 0

done = False

while not done:    

    # Select the action with the best Q-value
    action = np.argmax(q_table[state])

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