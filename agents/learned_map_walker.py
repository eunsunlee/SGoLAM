import argparse
import os
import random
# import sys
# sys.path.insert(0, "")
import numpy as np
import habitat
from habitat.core.challenge import Challenge
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainerO, PPOTrainerNO
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config    
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
import torch
from tqdm import tqdm
import re


class LearnedMapWalker(habitat.Agent):
    def __init__(self, exp_config, challenge, checkpoint_path=None):
        self._POSSIBLE_ACTIONS = np.array([0,1,2,3])
        opts = []
        self.num_obs = 1 # Batch dimension of observation

        config = get_config(exp_config, opts)  # exp_config is the config contained in habitat_baselines
        random.seed(config.TASK_CONFIG.SEED)
        np.random.seed(config.TASK_CONFIG.SEED)

        config.defrost()
        config.RL.PPO.hidden_size = 512

        if checkpoint_path is not None:
            config.EVAL_CKPT_PATH_DIR = checkpoint_path
        
        # Account for single GPU case
        if torch.cuda.device_count() == 1:
            config.TORCH_GPU_ID = 0

        config.freeze()

        self.trainer = PPOTrainerNO(config)
        self.trainer.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.trainer.load_checkpoint(config.EVAL_CKPT_PATH_DIR, map_location="cpu")

        if self.trainer.config.EVAL.USE_CKPT_CONFIG:
            config = self.trainer._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.trainer.config.clone()

        self.ppo_cfg = config.RL.PPO

        self.trainer.envs = challenge._env
        self.trainer._setup_actor_critic_agent(self.ppo_cfg)

        self.trainer.agent.load_state_dict(ckpt_dict["state_dict"])
        self.trainer.actor_critic = self.trainer.agent.actor_critic

        self.test_recurrent_hidden_states = torch.zeros(
            self.trainer.actor_critic.net.num_recurrent_layers,
            self.num_obs,
            self.ppo_cfg.hidden_size,
            device=self.trainer.device,
        )
        self.test_global_map = torch.zeros(
            self.num_obs,
            self.trainer.config.RL.MAPS.global_map_size,
            self.trainer.config.RL.MAPS.global_map_size,
            self.trainer.config.RL.MAPS.global_map_depth,
        )

        self.prev_actions = torch.zeros(
            self.num_obs, 1, device=self.trainer.device, dtype=torch.long
        )

        self.not_done_masks = torch.zeros(
            self.num_obs, 1, device=self.trainer.device
        )

        self.trainer.actor_critic.eval()
        self.stats_episodes = dict()
        
        if challenge.phase is None or challenge.phase == "dev":
            total_episodes = challenge.num_episodes
        else:
            total_episodes = len(challenge._env.episodes)
        self.pbar = tqdm(total=total_episodes, disable=(challenge.phase == 'standard' or challenge.phase == "challenge"))
        self.found_call = 0
        self.challenge = challenge
        self.N_ON = int(re.search(r'\d+', challenge._env._config.DATASET.DATA_PATH).group())  # Extract number of objects to follow from DATA_PATH

    def reset(self):
        self.pbar.update()
        self.test_recurrent_hidden_states = torch.zeros(
            self.trainer.actor_critic.net.num_recurrent_layers,
            self.num_obs,
            self.ppo_cfg.hidden_size,
            device=self.trainer.device,
        )

        self.test_global_map = torch.zeros(
            self.num_obs,
            self.trainer.config.RL.MAPS.global_map_size,
            self.trainer.config.RL.MAPS.global_map_size,
            self.trainer.config.RL.MAPS.global_map_depth,
        )

        self.prev_actions = torch.zeros(
            self.num_obs, 1, device=self.trainer.device, dtype=torch.long
        )

        self.not_done_masks = torch.zeros(
            self.num_obs, 1, device=self.trainer.device
        )

        self.found_call = 0
        pass

    def act(self, observations):
        for key in observations.keys():
            observations[key] = torch.tensor(observations[key], device=self.trainer.device).unsqueeze(0)
        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
                self.test_global_map,
            ) = self.trainer.actor_critic.act(
                observations,
                self.test_recurrent_hidden_states,
                self.test_global_map,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
        if actions.item() == 0:
            self.found_call += 1
        
        if self.found_call == self.N_ON:
            self.challenge._env._task.is_stop_called = True
            
        self.not_done_masks.fill_(1.0)

        self.prev_actions.fill_(actions.item())

        return {"action": actions.item()}

def get_agent(exp_config, challenge, checkpoint_path=None):
    return LearnedMapWalker(exp_config, challenge, checkpoint_path)
