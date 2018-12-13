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
from ray.tune.logger import UnifiedLogger
from fed_utils import *

from pathlib import Path
import os
import tempfile
import time
import uuid

REWARD, NAIVE, INDEPENDENT, MAX = 'reward', 'naive', 'independent', 'max'
CHOICES = [REWARD, NAIVE, INDEPENDENT, MAX]
parser = argparse.ArgumentParser()
parser.add_argument("--num_agents", '-na', type=int, default=10)
parser.add_argument("--num_iters", '-iters', type=int, default=25)
parser.add_argument('--strategy', '-s', type=str, choices=CHOICES, default=INDEPENDENT)
parser.add_argument('--timesteps_per_iteration', '-tsteps', type=int, default=1000)
parser.add_argument('--target_network_update_freq', '-target_freq', type=int, default=500)
parser.add_argument('--env', '-e', type=str, default="CartPole-v0")
parser.add_argument('--num_samples', '-ns', type=int, default=1)
parser.add_argument('--name', type=str, default="fed_experiment")
parser.add_argument('--comm', '-c', type=float, default=0.05)

def reset_adam(agent):
    with agent.local_evaluator.tf_sess.graph.as_default():
        sess = agent.local_evaluator.tf_sess
        sess.run(agent.reset_adam_optimizer)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    # Increasing the comm round length shouldn't just strictly make training better
    if args.comm:
        default_timesteps_per_iterations = args.timesteps_per_iteration
        default_target_network_update_freq = args.target_network_update_freq
        default_num_iters = args.num_iters
        args.timesteps_per_iteration = int(default_num_iters * default_timesteps_per_iterations * args.comm)
        args.target_network_update_freq = int(default_num_iters * default_target_network_update_freq * args.comm)
        args.num_iters = int(default_num_iters * default_timesteps_per_iterations / args.timesteps_per_iteration)
        print("Num-iters: {}".format(args.num_iters))
        print("Timesteps: {}".format(args.timesteps_per_iteration))

    config = {"args": args}

    exp_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        args.name,
        args.env,
        args.num_agents,
        args.num_iters,
        args.strategy,
        args.timesteps_per_iteration,
        args.target_network_update_freq,
        args.num_samples,
        args.comm,
        time.strftime("%Y-%m-%d_%H-%M-%S")
    )


    def FED_RL(config, reporter):
        args = config["args"]
        date_suffix = time.strftime("%Y-%m-%d_%H-%M-%S")

        def logger_will(config):
            path = Path.home() / "ray_results" / exp_name / "agents" / str(date_suffix)
            logdir_prefix = "AGENT"
            if not os.path.exists(str(path)):
                os.makedirs(str(path))
            logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=str(path))
            return UnifiedLogger(config, logdir, None)
         
        agents = [DQNAgent(
            env=args.env,
            config={
                "gamma": 0.95,
                "n_step": 3, # is this neccesary? Default is 1
                "timesteps_per_iteration": args.timesteps_per_iteration,
                "target_network_update_freq": args.target_network_update_freq,
            },
            logger_creator=logger_will,
        ) for _ in range(args.num_agents)]

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
                print("reporting results for:", a)
                results.append(result)
                # reset adam
                if args.strategy != INDEPENDENT:
                    reset_adam(a)

            avg_returns = [result['episode_reward_mean'] for result in results]

            # if INDEPENDENT, agent weights do not need to be modified
            if args.strategy != INDEPENDENT:
                # gather all weights
                all_weights = [a.get_weights(["default"]).get("default") for a in agents]

                if args.strategy == NAIVE:
                    new_weights = compute_average_weights(all_weights)
                elif args.strategy == REWARD:
                    new_weights = compute_reward_weighted_avg_weights(all_weights, avg_returns)
                elif args.strategy == MAX:
                    new_weights = compute_max_reward_weights(all_weights, avg_returns)
                
                [a.set_weights({"default": new_weights}) for a in agents]
            reporter(avg_agent_reward=np.mean(avg_returns), std_agent_reward=np.std(avg_returns))

    configuration = tune.Experiment(
        args.name,
        run=FED_RL,
        trial_resources={"cpu": 12},
        stop={},  
        config=config,
        num_samples=args.num_samples,
        local_dir=str(Path.home() / "ray_results" / exp_name)
    )

    trials = tune.run_experiments(configuration, verbose=False)

