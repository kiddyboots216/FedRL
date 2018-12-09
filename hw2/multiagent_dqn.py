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
from fed_utils import *

REWARD, NAIVE, INDEPENDENT, MAX = 'reward', 'naive', 'independent', 'max'
CHOICES = [REWARD, NAIVE, INDEPENDENT, MAX]
parser = argparse.ArgumentParser()
parser.add_argument("--num-trainers", '-nt', type=int, default=2)
parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument('--strategy', '-s', type=str, choices=CHOICES, default=INDEPENDENT)
parser.add_argument('--timesteps_per_iteration', '-tsteps', type=int, default=1000)
parser.add_argument('--target_network_update_freq', '-target_freq', type=int, default=500)
parser.add_argument('--env', '-e', type=str, default="CartPole-v0")
parser.add_argument('--comm', '-c', type=float, default=0.05)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    # Increasing the comm round length shouldn't just strictly make training better
    if args.comm:
        args.timesteps_per_iteration = args.num_iters * args.timesteps_per_iteration * args.comm
        args.target_network_update_freq = args.num_iters * args.target_network_update_freq * args.comm

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
        "timesteps_per_iteration": args.timesteps_per_iteration,
        "target_network_update_freq": args.target_network_update_freq,
    }) for _ in range(args.num_trainers)]

    t = trainers[0]
    new_weights = t.get_weights(["dqn_policy"]).get("dqn_policy")
    [t.set_weights({"dqn_policy": new_weights}) for t in trainers]
    for i in range(args.num_iters):
        print("== Iteration", i, "==")
        results = []
        # Improve each Agent
        for trainer in trainers:
            print("-- {} --".format(trainer._agent_name))
            result = trainer.train()
            print(pretty_print(result))
            results.append(result)
        
        # if INDEPENDENT, agemnt weights do not need to be modified
        if args.strategy != INDEPENDENT:
            # gather all weights
            all_weights = [t.get_weights(["dqn_policy"]).get("dqn_policy") for t in trainers]

            avg_returns = [result['episode_reward_mean'] for result in results]
            if args.strategy == NAIVE:
                new_weights = compute_average_weights(all_weights)
            elif args.strategy == REWARD:
                new_weights = compute_reward_weighted_avg_weights(all_weights, avg_returns)
            elif args.strategy == MAX:
                new_weights = compute_max_reward_weights(all_weights, avg_returns)
            
            [t.set_weights({"dqn_policy": new_weights}) for t in trainers]
