import gym
import numpy as np
from easydict import EasyDict

import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv

def make_multiagent(args):
    class MultiEnv(MultiAgentEnv):
        def __init__(self):
            self.agents = [gym.make(args.env) for _ in range(args.num_agents)]
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

def make_fed_env(args):   
    FedEnv = make_multiagent(args)
    env_name = "multienv_FedRL"
    register_env(env_name, lambda _: FedEnv())
    return env_name

def gen_policy_graphs(args):
    single_env = gym.make(args.env)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    policy_graphs = {f'agent_{i}': (None, obs_space, act_space, {}) 
         for i in range(args.num_agents)}
    return policy_graphs

def policy_mapping_fn(agent_id):
    return f'agent_{agent_id}'
def change_weights(weights, i):
    """
    Helper function for FedQ-Learning
    """
    dct = {}
    for key, val in weights.items():
        # new_key = key
        still_here = key[:6]
        there_after = key[7:]
        # new_key[6] = i
        new_key = still_here + str(i) + there_after
        dct[new_key] = val
    # print(dct.keys())
    return dct

def synchronize(agent, weights, num_agents):
    """
    Helper function to synchronize weights of the multiagent
    """
    weights_to_set = {f'agent_{i}': weights 
         for i in range(num_agents)}
    # weights_to_set = {f'agent_{i}': change_weights(weights, i) 
    #    for i in range(num_agents)}
    # print(weights_to_set)
    agent.set_weights(weights_to_set)

def uniform_initialize(agent, num_agents):
    """
    Helper function for uniform initialization
    """
    new_weights = agent.get_weights(["agent_0"]).get("agent_0")
    # print(new_weights.keys())
    synchronize(agent, new_weights, num_agents)

def compute_softmax_weighted_avg(weights, alphas, num_agents, temperature=1):
    """
    Helper function to compute weighted avg of weights weighted by alphas
    Weights and alphas must have same keys. Uses softmax.
    params:
        weights - dictionary
        alphas - dictionary
    returns:
        new_weights - array
    """
    def softmax(x, beta=temperature, length=num_agents):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(beta * (x - np.max(x)))
        return (e_x / e_x.sum()).reshape(length, 1)
    
    alpha_vals = np.array(list(alphas.values()))
    soft = softmax(alpha_vals)
    weight_vals = np.array(list(weights.values()))
    new_weights = sum(np.multiply(weight_vals, soft))
    return new_weights

def reward_weighted_update(agent, result, num_agents):
    """
    Helper function to synchronize weights of multiagent via
    reward-weighted avg of weights
    """
    return softmax_reward_weighted_update(agent, result, num_agents, temperature=0)

def softmax_reward_weighted_update(agent, result, num_agents, temperature=1):
    """
    Helper function to synchronize weights of multiagent via
    softmax reward-weighted avg of weights with specific temperature
    """
    all_weights = agent.get_weights()
    policy_reward_mean = result['policy_reward_mean']
    episode_reward_mean = result['episode_reward_mean']
    if policy_reward_mean:
        new_weights = compute_softmax_weighted_avg(all_weights, policy_reward_mean, num_agents, temperature=temperature)
        synchronize(agent, new_weights, num_agents)

def fed_train(args):
    temp_schedule = args.temp_schedule
    temperature = temp_schedule[0]
    hotter_temp = temp_schedule[1]
    temp_shift = temp_schedule[2]
    fed_schedule = args.fed_schedule
    num_iters = fed_schedule[0]
    increased_iters = fed_schedule[1]
    fed_shift = fed_schedule[2]
    
    num_agents = args.num_agents
    def fed_learn(info):
#       get stuff out of info
        result = info["result"]
        agent = info["trainer"]
        optimizer = agent.optimizer
        if result['timesteps_total'] > fed_shift:
            num_iters = increased_iters
        if result['timesteps_total'] > temp_shift:
            temperature = hotter_temp
        # correct result reporting
        result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
        result['episode_reward_max'] = result['episode_reward_max']/num_agents
        result['episode_reward_min'] = result['episode_reward_min']/num_agents
        result['federated'] = "No federation"
        if result['training_iteration'] == 1:
            uniform_initialize(agent, num_agents)
        elif result['training_iteration'] % num_iters == 0:
            result['federated'] = f"Federation with {temperature}"
            # update weights
            softmax_reward_weighted_update(agent, result, num_agents, temperature)
            # clear buffer, don't want smoothing here
            optimizer.episode_history = []
    return fed_learn

def fedrl(args):
    ray.init(ignore_reinit_error=True)
    policy_graphs = gen_policy_graphs(args)
    multienv_name = make_fed_env(args)
    callback = fed_train(args)
    tune.run(
        args.algo,
        name=f"{args.env}-{args.algo}-{args.num_agents}",
        stop={"episode_reward_mean": 9800},
        config={
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(lambda agent_id: f'agent_{agent_id}'),
                },
                "env": multienv_name,
                "gamma": 0.99,
                "lambda": 0.95,
                "kl_coeff": 1.0,
                "num_sgd_iter": 32,
                "lr": .0003 * args.num_agents,
                "vf_loss_coeff": 0.5,
                "clip_param": 0.2,
                "sgd_minibatch_size": 4096,
                "train_batch_size": 65536,
                "grad_clip": 0.5,
                "batch_mode": "truncate_episodes",
                "observation_filter": "MeanStdFilter",
                # "lr": tune.grid_search(args.lrs),
#                 "simple_optimizer": True,
                "callbacks":{
                    "on_train_result": tune.function(callback),
                },
                "num_workers": args.num_workers,
                "num_gpus": 1,
            },
        checkpoint_at_end=True
    )


args = EasyDict({
    'num_agents': 5,
    'num_workers': 7,
    'fed_schedule': [1, 5, 2e7],
    # 'temperatures': [0, 8, 0.5, 4, 2, 1, 16],
    'temp_schedule': [0.5, 2, 2e7],
    # 'timesteps': 1e7,
    # 'lr': 5e-4,
    # 'lrs': [5e-5, 5e-4, 5e-3],
    # 'episodes': 150,
#     'num_iters': 100,
    'env': 'HalfCheetah-v2',
    'name': 'fed_experiment',
    'algo': 'PPO',
})
# train
fedrl(args)
# eval