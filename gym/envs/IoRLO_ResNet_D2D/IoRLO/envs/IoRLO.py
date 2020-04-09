import gym
import numpy as np

from . import per_block_latency as pbl
from . import per_block_energy as pbe

from . import tool

# 17 for ResNet-32
N_IN = 17  # input dimension
N_OUT = 17  # output dimension

IOF = 300  # discounting factor between latency and energy

IFDONE = 0  # for quick done defined by third-party (this will never happen)

class IoRLO(gym.Env):

    def __init__(self):
        self.n_features = N_IN
        self.n_actions = N_OUT
        self.state = np.zeros(N_IN)
        self.counts = 0

    def step(self, action):
        """
        :param action:
        :return ob, reward, episode_over, info: tuple
            ob (object):
                an environment-specific object representing your observation of the environment.
            reward (float):
                amount of reward achieved by the previous action. The scale varies between environments, but the goal
                is always to increase your total reward.
            episode_over (bool):
                whether it is time to reset the environment again. Most (but not all) tasks are divided up into well-
                defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the
                pole tipped too far, or you lost your last life).
            info (dict):
                diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it
                might contain the raw probabilities behind the environment's last state change). However, official
                evaluations of your agent are not allowed to use this for learning.
        """

        # with open('action.txt', 'a') as f:
        #     f.write(str(action) + '\n')

        # get state (computing & communication cost) according to the action
        
        # ResNet-32
        state_pos = 0
        for item in action:
            if item == 0:
                self.state[state_pos] = pbl.res32_latency_T()[state_pos] + IOF * pbe.res32_energy_T()[state_pos]
            else: # item == 1
                self.state[state_pos] = pbl.res32_latency_H_h()[state_pos]
                # self.state[state_pos] = pbl.res32_latency_H_l()[state_pos]
            state_pos += 1

        # Note: computing cost (= latency + energy)
        comp_cost = np.sum(self.state)

        # communication cost
        comm_cost_latency, comm_cost_energy = tool.comm_cost_res32_D2D(action, N_OUT)  # res32_D2D
        comm_cost = comm_cost_latency + IOF * comm_cost_energy
        # print('comm_cost_latency: ', comm_cost_latency)
        # print('comm_cost_energy: ', comm_cost_energy)

        # total cost
        total_cost = comp_cost + comm_cost

        # reward
        reward = -total_cost
        # print('reward: ', reward)

        self.counts += 1

        done = True if reward > IFDONE else False

        self.state = tool.norm_array(self.state)
        return self.state, reward, done, {}

    def reset(self):
        # init all the blocks are exec on the mobile device
        self.state = pbl.res32_latency_T() + IOF * pbe.res32_energy_T()
        # print('self.state: ', self.state)
        self.counts = 0

    def render(self):
        return None

    def close(self):
        return None
