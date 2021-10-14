import habitat
import os
from habitat_baselines.common.utils import generate_video
from habitat.utils.visualizations.utils import observations_to_image_challenge, observations_to_image_custum
import numpy as np


class VideoWalker(habitat.Agent):
    """
    Wrapper class for agents that will automatically log videos per episode.
    """
    def __init__(self, Walker: habitat.Agent, video_dir: str):
        super().__init__()
    
        if video_dir is not None:
            os.makedirs(video_dir, exist_ok=True)
        self.walker = Walker

        self.rgb_frames = [[] for _ in range(self.walker.num_obs)]
        self.video_dir = video_dir
        self.episode_id = 0
    
    def reset(self):
        if self.walker.challenge._env.episode_over:
            for i in range(self.walker.num_obs):
                generate_video(
                        video_option=["disk"],
                        video_dir=self.video_dir,
                        images=self.rgb_frames[i],
                        episode_id=self.episode_id,
                        checkpoint_idx=0,
                        tb_writer=None,
                        metrics={}
                )

            self.rgb_frames = [[] for _ in range(self.walker.num_obs)]
            self.episode_id += 1
            self.walker.reset()

    def act(self, observations):
        action = self.walker.act(observations)
        action_val = np.array([action["action"]])
        observations = [observations]

        for i in range(self.walker.num_obs):
            if getattr(self.walker, 'visualize_list', None):
                frame = observations_to_image_custum(observations[i], self.walker, action_val)
            else:
                frame = observations_to_image_challenge(observations[i], 
                    {'feature_map': getattr(self.walker, 'feature_map', None), 'occupancy_map': getattr(self.walker, 'occupancy_map', None), \
                    'agent_pos': getattr(self.walker, 'agent_pos', None)}, action_val)
            if len(frame.shape) == 4:
                frame = frame.squeeze()
            self.rgb_frames[i].append(frame)

        return action
