import gym
import numpy as np

import ray
from ray.rllib.agents.pg.pg import PGAgent
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.trainable import Trainable
from ray import tune

import gym 

REWARD, NAIVE, INDEPENDENT, MAX = 'reward', 'naive', 'independent', 'max'
CHOICES = [REWARD, NAIVE, INDEPENDENT, MAX]

from easydict import EasyDict

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym

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
	policy_graphs = {f'agent_{i}': (args.graph_type, obs_space, act_space, {}) 
		 for i in range(args.num_agents)}
	return policy_graphs

def policy_mapping_fn(agent_id):
	return f'agent_{agent_id}'
def change_weights(weights, i):
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
	# 	 for i in range(num_agents)}
	# print(weights_to_set)
	agent.set_weights(weights_to_set)

def uniform_initialize(agent, num_agents):
	"""
	Helper function for uniform initialization
	"""
	new_weights = agent.get_weights(["agent_0"]).get("agent_0")
	# print(new_weights.keys())
	synchronize(agent, new_weights, num_agents)
	
def compute_weighted_avg(weights, alphas):
	"""
	Helper function to compute weighted avg of weights weighted by alphas
	Weights and alphas must have same keys
	params:
		weights - dictionary
		alphas - dictionary
	returns:
		new_weights - array
	"""
	temp = np.array([np.multiply(weights[c], alphas[c]) for c in weights.keys()])
	summed_weights = sum(temp)
	summed_rewards = sum(alphas.values())
	new_weights = np.divide(summed_weights, summed_rewards)
	return new_weights
def compute_softmax_weighted_avg(weights, alphas):
	"""
	Helper function to compute weighted avg of weights weighted by alphas
	Weights and alphas must have same keys. Uses softmax.
	params:
		weights - dictionary
		alphas - dictionary
	returns:
		new_weights - array
	"""
	temp = np.array([np.multiply(weights[c], np.exp(alphas[c])) for c in weights.keys()])
	summed_weights = sum(temp)
	summed_rewards = sum([np.exp(val) for val in alphas.values()])
	new_weights = np.divide(summed_weights, summed_rewards)
	return new_weights
def reward_weighted_update(agent, result, num_agents):
	"""
	Helper function to synchronize weights of multiagent via
	reward-weighted avg of weights
	"""
	all_weights = agent.get_weights()
	policy_reward_mean = result['policy_reward_mean']
	new_weights = compute_weighted_avg(all_weights, policy_reward_mean)
	synchronize(agent, new_weights, num_agents)

def softmax_reward_weighted_update(agent, result, num_agents):
	all_weights = agent.get_weights()
	policy_reward_mean = result['policy_reward_mean']
	episode_reward_mean = result['episode_reward_mean']
	try:
		new_weights = compute_softmax_weighted_avg(all_weights, policy_reward_mean)
		synchronize(agent, new_weights, num_agents)
	except:
		print(f"Couldn't update, probably because episode_reward_mean is {episode_reward_mean}")
		

class FedRLActor(Trainable):
	def __init__(self, config=None, logger_creator=None):
		super().__init__(config, logger_creator)
		agent_config = config["for_agent"]
		algo_config = config["for_algo"]
		self.agent = algo_config["agent_type"](config=agent_config)
		self.num_agents = len(agent_config["multiagent"]["policy_graphs"].keys())
		uniform_initialize(self.agent, self.num_agents)
	def _train(self, ):
		result = self.agent.train()
		# modify reporting for multiagent a bit ONLY when same MDP
		result['episode_reward_mean'] = result['episode_reward_mean']/self.num_agents
		result['episode_reward_max'] = result['episode_reward_max']/self.num_agents
		result['episode_reward_min'] = result['episode_reward_min']/self.num_agents
		# reporter(**result)
		print(pretty_print(result))
		# Do update
		# if result['episodes_total'] > 5:
		#     if strategy == REWARD:
		softmax_reward_weighted_update(self.agent, result, self.num_agents)

		# reward_weighted_update(self.agent, result, self.num_agents)
		# print("finished reward weighted update")
		return result
	def _save(self, checkpoint_dir=None):
		return self.agent.save(checkpoint_dir)
	def _restore(self, checkpoint):
		return self.agent.restore(checkpoint)

slow_start = True

def manage_curriculum(info):
	global slow_start
	print("Manage Curriculum callback called on phase {}".format(slow_start))
	result = info["result"]
	if slow_start and result["training_iteration"] % 100 == 0 and result["training_iteration"] != 0:
		slow_start = False
		agent = info["agent"]
		agent.optimizer.train_batch_size *= 5

def fed_learning(args):
	ray.init(ignore_reinit_error=True)
	policy_graphs = gen_policy_graphs(args)
	multienv_name = make_fed_env(args)
	tune.run(
		FedRLActor,
		name=f"{args.env}-{args.agent_type}-{args.num_agents}",
		stop={"training_iteration": 100},
		config={
			"for_agent": {
				"multiagent": {
					"policy_graphs": policy_graphs,
					"policy_mapping_fn": tune.function(lambda agent_id: f'agent_{agent_id}'),
					# "policies_to_train":
				},
				# "train_batch_size": 4000 * args.num_agents,
				"env": multienv_name,
				# "callbacks":{
				#     "on_train_result": tune.function(manage_curriculum),
				# },
				"num_workers": 5,
				# "num_gpus_per_worker": 0.3,
				"num_cpus_per_worker": 0.5,
			},
			"for_algo": {
			   # "num_iters": args.num_iters,
				"strategy": args.strategy,
				"agent_type": args.agent_type
			},
		},
		resources_per_trial={
			"extra_gpu": 0,
			"cpu": 1.0,
			"extra_cpu": 3.0,
		},
		checkpoint_at_end=True
	)


# print(e)

args = EasyDict({
	'num_agents': 5,
	'strategy': REWARD,
	'num_iters': 100,
	'env': 'MountainCarContinuous-v0',
	'name': 'fed_experiment',
	# 'agent_type': ray.rllib.agents.ddpg.ddpg.DDPGAgent,
	# 'graph_type': ray.rllib.agents.ddpg.ddpg_policy_graph.DDPGPolicyGraph,
	'agent_type': ray.rllib.agents.ppo.ppo.PPOAgent,
	'graph_type': ray.rllib.agents.ppo.ppo_policy_graph.PPOPolicyGraph,
	# 'graph_type': PGPolicyGraph,
#     'graph_type': ray.rllib.agents.a3c.a3c_tf_policy_graph.A3CPolicyGraph,
#     'agent_type': ray.rllib.agents.a3c.a3c.A3CAgent,
	# 'agent_type': PGAgent,
})
# train
fed_learning(args)
# eval