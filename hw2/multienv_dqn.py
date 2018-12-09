from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gym
import numpy as np
import tensorflow as tf

import ray
from ray import tune
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.test.test_multi_agent_env import MultiCartpole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from fed_utils import *

REWARD, NAIVE, INDEPENDENT, MAX = 'reward', 'naive', 'independent', 'max'
CHOICES = [REWARD, NAIVE, INDEPENDENT, MAX]
parser = argparse.ArgumentParser()
parser.add_argument("--num_agents", '-na', type=int, default=2)
parser.add_argument("--num_iters", '-iters', type=int, default=20)
parser.add_argument('--strategy', '-s', type=str, choices=CHOICES, default=INDEPENDENT)
parser.add_argument('--timesteps_per_iteration', '-tsteps', type=int, default=1000)
parser.add_argument('--target_network_update_freq', '-target_freq', type=int, default=500)
parser.add_argument('--env', '-e', type=str, default="CartPole-v0")

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    agents = [DQNAgent(
    env=args.env, 
    config={
        "gamma": 0.95,
        "n_step": 3, # is this neccesary? Default is 1
        "timesteps_per_iteration": args.timesteps_per_iteration,
        "target_network_update_freq": args.target_network_update_freq,
    }) for _ in range(args.num_agents)]

    new_weights = agents[0].get_weights(["default"]).get("default")
    [a.set_weights({"default": new_weights}) for a in agents]

    # add reset op for each agent    
    for a in agents:
        with a.local_evaluator.tf_sess.graph.as_default():
            opt = a.local_evaluator.policy_map['default']._optimizer
            a.reset_adam_optimizer = tf.variables_initializer(opt.variables())
            
    for i in range(args.num_iters):
        print("== Iteration", i, "==")
        results = []
        # Improve each Agent
        for a in agents:
            print("-- {} --".format(a._agent_name))
            result = a.train()
            print(pretty_print(result))
            results.append(result)
            # reset adam
            with a.local_evaluator.tf_sess.graph.as_default():
                sess = a.local_evaluator.tf_sess
                sess.run(a.reset_adam_optimizer)

        # if INDEPENDENT, agemnt weights do not need to be modified
        if args.strategy != INDEPENDENT:
            # gather all weights
            all_weights = [a.get_weights(["default"]).get("default") for a in agents]

            avg_returns = [result['episode_reward_mean'] for result in results]
            if args.strategy == NAIVE:
                new_weights = compute_average_weights(all_weights)
            elif args.strategy == REWARD:
                new_weights = compute_reward_weighted_avg_weights(all_weights, avg_returns)
            elif args.strategy == MAX:
                new_weights = compute_max_reward_weights(all_weights, avg_returns)
            
            [a.set_weights({"default": new_weights}) for a in agents]
