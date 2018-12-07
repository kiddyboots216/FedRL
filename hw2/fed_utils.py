import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

def compute_average_weights(all_weights):
    averaged = []
    number_of_variables = len(all_weights[0])
    # iterate through each model layer and average them
    for i in range(number_of_variables):
        averaged.append(np.mean([weights[i] for weights in all_weights], axis=0))
    return np.asarray(averaged)
    
def compute_reward_weighted_avg_weights(all_weights, avg_returns):
    temp = np.array([np.multiply(all_weights[c], avg_returns[c]) for c in range(len(all_weights))])
    summed_weights = sum(temp)
    summed_rewards = sum(avg_returns)
    return np.divide(summed_weights, summed_rewards)

def compute_max_reward_weights(all_weights, avg_returns):
    return all_weights[np.argmax(avg_returns)]

def make_multiagent(env_name):
    class MultiEnv(MultiAgentEnv):
        def __init__(self, num):
            self.agents = [gym.make(env_name) for _ in range(num)]
            self.dones = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space

        def reset(self):
            self.dones = set()
            return {i: a.reset() for i, a in enumerate(self.agents)}

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info

    return MultiEnv    