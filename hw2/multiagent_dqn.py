from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Example of using two different training methods at once in multi-agent.
Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. We periodically sync weights
between the two trainers (note that no such syncing is needed when using just
a single training method).
For a simpler example, see also: multiagent_cartpole.py
"""

import argparse
import gym
import numpy as np

import ray
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.test.test_multi_agent_env import MultiCartpole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()
parser.add_argument("--num-trainers", type=int, default=2)
parser.add_argument("--num-iters", type=int, default=20)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    # Simple environment with 1 cartpole
    register_env("multi_cartpole", lambda _: MultiCartpole(1))
    single_env = gym.make("CartPole-v0")
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    
    policy_graphs = {
        "dqn_policy": (DQNPolicyGraph, obs_space, act_space, {})
    }
    def policy_mapping_fn(agent_id):
        return "dqn_policy"

    trainers = [DQNAgent(
    env="multi_cartpole",
    config={
        "multiagent": {
            "policy_graphs": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["dqn_policy"],
        },
        "gamma": 0.95,
        "n_step": 3,
    }) for _ in range(args.num_trainers)]

    def compute_average_weights(all_weights):
        all_weights = [weight.get("dqn_policy") for weight in all_weights]
        averaged = []
        number_of_variables = len(all_weights[0])
        # iterate through each model layer and average them
        for i in range(number_of_variables):
            averaged.append(np.mean([weights[i] for weights in all_weights], axis=0))
        return np.asarray(averaged)
    
    def compute_reward_weighted_avg_weights(all_weights, avg_returns):
        all_weights = [weight.get("dqn_policy") for weight in all_weights]
        temp = np.array([np.multiply(all_weights[c], avg_returns[c]) for c in range(args.num_trainers)])
        summed_weights = sum(temp)
        summed_rewards = sum(avg_returns)
        return np.divide(summed_weights, summed_rewards)

    def compute_max_reward_weights(all_weights, avg_returns):
        all_weights = [weight.get("dqn_policy") for weight in all_weights]
        return all_weights[np.argmax(avg_returns)]

    for i in range(args.num_iters):
        print("== Iteration", i, "==")
        results = []
        # Improve each Agent
        for trainer in trainers:
            print("-- {} --".format(trainer._agent_name))
            result = trainer.train()
            print(pretty_print(result))
            results.append(result)
        # gather all weights
        all_weights = [t.get_weights(["dqn_policy"]) for t in trainers]
        # compute average weight
        # new_weights = compute_average_weights(all_weights)
        avg_returns = [result['episode_reward_mean'] for result in results]
        # new_weights = compute_reward_weighted_avg_weights(all_weights, avg_returns)
        new_weights = compute_max_reward_weights(all_weights, avg_returns)
        # set weights of all agents
        [t.set_weights({"dqn_policy": new_weights}) for t in trainers]