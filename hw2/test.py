import ray
import ray.tune as tune

ray.init()
tune.run_experiments({
    "my_experiment": {
        "run": "PPO",
        "env": "CartPole-v0",
        "stop": {"episode_reward_mean": 200},
        "num_samples": 3,
        "config": {
            "num_gpus": 0,
            "num_workers": 1,
            # "sgd_stepsize": tune.grid_search([0.01, 0.001, 0.0001]),
        },
    },
})