from subprocess import call
import numpy as np

# comm = [0.005, 0.01, 0.02, 0.05, 0.1]
comm = [0.01]

for c in reversed(comm):
    # call(["python", "multienv_dqn.py", "--comm", str(c), "-na", "10", "-e", "CartPole-v1", "-s", "reward", "-ns", "3"])
    call(["python", "multiagent_dqn.py", "--comm", str(c), "-na", "10", "-e", "CartPole-v1", "-s", "reward"])
