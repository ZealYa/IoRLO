import tensorflow as tf
import numpy as np
import gym
import time

import algorithm.DDPG_u2 as DDPG_u2
# import algorithm.DDPG as DDPG
import IoRLO
import IoRLO.envs.tool as tool


MAX_EPISODES = 200
MAX_EP_STEPS = 200
RENDER = False
# ENV_NAME = 'IoRLO-v0'
MEMORY_CAPACITY = 10000  # 10000

N_IN = 10
N_OUT = 10
ThresC = 0.4

# env = gym.make(ENV_NAME)
env = gym.make('IoRLO-v0')
env = env.unwrapped
env.seed(1)

s_dim = N_IN
a_dim = N_OUT
a_bound = 1

ddpg = DDPG_u2.DDPG(a_dim, s_dim, a_bound)
# ddpg = DDPG.DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration  3
start_time = time.time()

for i in range(MAX_EPISODES):

    # s = env.reset()
    env.reset()
    s = env.state
    ep_reward = 0
    
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.random.normal(a, var)
        tool.apply_action_D2D(a, N_OUT, ThresC)  # D2D
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9998 # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            with open('action.txt', 'a') as f:
                f.write(str(a) + '\n')
            with open('ep_reward.txt', 'a') as f:
                f.write(str(ep_reward/MAX_EP_STEPS) + '\n')
            print('Episode:', i, ' Reward: %i' % int(ep_reward/MAX_EP_STEPS), 'Explore: %.2f' % var)
            # if ep_reward > -300:RENDER = True
            break

print('Running time: ', time.time() - start_time)
