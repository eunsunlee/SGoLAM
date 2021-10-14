from random import Random
import numpy as np
import habitat


"""
In order to use evaluate.py, please implement the following:
1. A walker that inherits habitat.Agent.
2. Walker must have a reset() and act() method.
3. get_agent function that returns a walker instance with the same function signature as below.
"""


class ExampleWalker(habitat.Agent):
    def __init__(self):
        self._POSSIBLE_ACTIONS = np.array([0,1,2,3])

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self._POSSIBLE_ACTIONS)}


def get_agent(exp_config, challenge, checkpoint_path=None):
    return ExampleWalker()
