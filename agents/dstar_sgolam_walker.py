import argparse
import os
import random
import math
import numpy as np
from torch._C import device
import habitat
from habitat.core.challenge import Challenge
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
import torch
import torch.nn as nn
from tqdm import tqdm
import re
from queue import PriorityQueue
from utils.geometry import Projection, Registration, to_grid, GlobalMap2World
from utils.visualize import debug_visualize as dv
from utils.color import compute_hsv_metric_from_rgb
from utils.frontier import Frontier
from utils.astar import astar_algorithm
from utils.control import get_action
import skimage
import skimage.measure
from utils.planner import FMMPlanner
from utils.dstar import DifferentiableStarPlanner, generate_2dgrid
import torch.nn.functional as F


class DStarSGoLAMWalker(habitat.Agent):
    """
    Agent implementation of Simultaneous Goal Localization and Mapping (SGoLAM)
    """
    def __init__(self, exp_config, challenge, checkpoint_path=None):
        self._POSSIBLE_ACTIONS = np.array([0,1,2,3])
        self.num_obs = 1 # Batch dimension of observation
        self.config = get_config(exp_config)

        self.total_objects = self.config.TOTAL_OBJECTS  # Total number of objects that could be a goal
        self.map_size = self.config.AGENT.MAPS.map_size  # Height and width of map
        self.feature_map_channels = self.config.AGENT.MAPS.feature_map_channels  # Total number of channels in feature map
        self.occupancy_map_channels = self.config.AGENT.MAPS.occupancy_map_channels  # Total number of channels in occupancy map
        self.seed = self.config.SEED
        self.affinity_measure = self.config.AGENT.MAPS.affinity_measure  # How to measure affinity
        self.goal_measure = self.config.AGENT.MAPS.goal_measure  # How to measure is a goal is found in goal map
        self.goal_threshold = self.config.AGENT.MAPS.goal_threshold  # Threshold over which will be considered a goal
        self.egocentric_map_size = self.config.AGENT.MAPS.egocentric_map_size  # Size of egocentric map which will be registered after each agent step
        self.coordinate_min = self.config.AGENT.MAPS.coordinate_min  # Minimum coordinate global map can keep track of
        self.coordinate_max = self.config.AGENT.MAPS.coordinate_max  # Maximum coordinate global map can keep track of
        self.depth_march_size = self.config.AGENT.MAPS.depth_march_size  # March size for building occupancy map
        self.visualize_list = self.config.AGENT.VISUALIZE.target  # List containing name of attributes to visualize

        random.seed(self.seed)
        np.random.seed(self.seed)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = torch.device("cuda:1")
            else:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.goal_features = torch.zeros([self.total_objects, self.feature_map_channels], device=self.device)  # Average color of each goal
        self.goal_features = self.fill_features()

        self.goal_dict = self.fill_goal_dict()  # Maps goal name to goal index

        # For now, let's assume RGB values are projected to self.feature_map
        self.feature_map = torch.zeros(
            self.num_obs,
            self.map_size,
            self.map_size,
            self.feature_map_channels,
            device=self.device
        )

        self.frontier_map = torch.zeros(
            self.num_obs,
            self.map_size,
            self.map_size,
            self.feature_map_channels,
            device=self.device
        )

    
        # Occupancy map: 0.0 = unexplored, 0.5 = vacant space, 1.0 = occupied space
        self.occupancy_map = torch.zeros(
            self.num_obs,
            self.map_size,
            self.map_size,
            self.occupancy_map_channels,
            device=self.device
        )

        self.frontier_map = torch.zeros(
            self.num_obs,
            self.map_size,
            self.map_size,
            self.feature_map_channels,
            device=self.device
        )

        self.binary_occupied_map = torch.zeros(
            self.num_obs,
            self.map_size,
            self.map_size,
            1,
            device=self.device
        )

        # self.goal_map contains the grid-wise, object-wise feature affinity
        self.goal_map = torch.zeros(
            self.num_obs,
            self.map_size,
            self.map_size,
            self.total_objects,
            device=self.device
        )

        self.prev_actions = torch.zeros(
            self.num_obs, 1, device=self.device, dtype=torch.long
        ).fill_(-1)
        

        if challenge.phase is None or challenge.phase == "dev":
            total_episodes = getattr(challenge, 'num_episodes', len(challenge._env.episodes))
        else:
            total_episodes = len(challenge._env.episodes)
        self.pbar = tqdm(total=total_episodes, disable=(challenge.phase == 'standard' or challenge.phase == "challenge"))
        self.found_call = 0
        self.challenge = challenge
        self.N_ON = int(re.search(r'\d+', challenge._env._config.DATASET.DATA_PATH).group())  # Extract number of objects to follow from DATA_PATH
        self.agent_pos = (self.map_size // 2, self.map_size // 2)  # Agent position with respect to global map (H, W position)
        self.agent_rot =  0
        self.forward_step_size = self.map_size // 2 - float(to_grid(
            self.map_size, 
            self.coordinate_min, 
            self.coordinate_max
        ).get_grid_coords(torch.tensor([[self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE,0]]))[0][0])
        self.goal_position = torch.tensor([-1, -1], device=self.device)  # H, W position
        
        
        # Geometric functions
        self.feature_projection = Projection(self.egocentric_map_size, self.map_size, self.device, self.coordinate_min, self.coordinate_max, \
            -0.8, 0.0, 'mean')
        self.vacant_projection = Projection(self.egocentric_map_size, self.map_size, self.device, self.coordinate_min, self.coordinate_max, \
            -0.9, -0.8, 'max')
        self.occupied_projection = Projection(self.egocentric_map_size, self.map_size, self.device, self.coordinate_min, self.coordinate_max, \
            -0.6, 0.0, 'max')
        self.floor_occupied_projection = Projection(self.egocentric_map_size, self.map_size, self.device, self.coordinate_min, self.coordinate_max, \
            -5.0, -2.0, 'max')
        self.goal_projection = Projection(self.egocentric_map_size, self.map_size, self.device, self.coordinate_min, self.coordinate_max, \
            -0.8, 0.0, 'mean')
        self.feature_registration = Registration(self.egocentric_map_size, self.map_size, self.feature_map_channels, self.device, \
            self.coordinate_min, self.coordinate_max, self.num_obs)
        self.occupancy_registration = Registration(self.egocentric_map_size, self.map_size, self.occupancy_map_channels, self.device, \
            self.coordinate_min, self.coordinate_max, self.num_obs)
        self.goal_registration = Registration(self.egocentric_map_size, self.map_size, self.total_objects, self.device, \
            self.coordinate_min, self.coordinate_max, self.num_obs)
        self.world_projection = None  # World coordinate conversion function set at the beginning of each episode

        # Goal maps for visualization
        self.red_goal_map = self.goal_map[..., 0: 1]
        self.green_goal_map = self.goal_map[..., 1: 2]
        self.blue_goal_map = self.goal_map[..., 2: 3]
        self.yellow_goal_map = self.goal_map[..., 3: 4]
        self.white_goal_map = self.goal_map[..., 4: 5]
        self.pink_goal_map = self.goal_map[..., 5: 6]
        self.black_goal_map = self.goal_map[..., 6: 7]
        self.cyan_goal_map = self.goal_map[..., 7: 8]

        # Binary detection masks for visualization
        self.goal_detect_img = torch.zeros(self.num_obs, 275, 275, 8, device=self.device)
        self.red_detect_img = self.goal_detect_img[0, ..., 0: 1]
        self.green_detect_img = self.goal_detect_img[0, ..., 1: 2]
        self.blue_detect_img = self.goal_detect_img[0, ..., 2: 3]
        self.yellow_detect_img = self.goal_detect_img[0, ..., 3: 4]
        self.white_detect_img = self.goal_detect_img[0, ..., 4: 5]
        self.pink_detect_img = self.goal_detect_img[0, ..., 5: 6]
        self.black_detect_img = self.goal_detect_img[0, ..., 6: 7]
        self.cyan_detect_img = self.goal_detect_img[0, ..., 7: 8]

        # Ego feature view for visualization
        self.ego_img = torch.zeros(65, 65, 3)

        # Total step counter
        self.total_steps = 0

        # Total number of goals found
        self.goals_found = 0

        # Metrics
        self.recall = 0.0
        self.precision = 0.0

        # path found, collision
        self.path_found = False

        # DStar attributes
        self.planner = DifferentiableStarPlanner(device=self.device)
        self.coordinatesGrid = generate_2dgrid(self.map_size, self.map_size, False).to(
            self.device
        )
        self.planned2Dpath = []
        self.waypointPose2D = None  # H, W coordinate of next waypoint
        self.prev_agent_pos = self.agent_pos
        
        # Collision attributes
        self.COLLISION_PROTOCOL = False
        self.collision_step = 0

        # Thrashing attributes
        self.THRASHING_PROTOCOL = False
        self.thrashing_step = 0

        self.preprocessNet = nn.Conv2d(
            1, 1, kernel_size=(3, 3), padding=1, bias=False
        )
        self.preprocessNet.weight.data = torch.from_numpy(
            np.array(
                [
                    [
                        [
                            [0.00001, 0.0001, 0.00001],
                            [0.0001, 1, 0.0001],
                            [0.00001, 0.0001, 0.00001],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        )
        self.preprocessNet.to(self.device)

        self.agent_gps_pos = torch.zeros(2)
        self.prev_agent_gps_pos = torch.zeros(2)
        self.explore_cnt = 0

        self.action_sequence = []  # Saves all actions
        self.agent_pos_sequence = []  # Saves all agent positions as a list
        self.dont_go_list = []  # Saves all indices where the agent should not go

        # Invalid RGBD attributes
        self.valid_obs = True
        self.INVALID_RGBD_PROTOCOL = False
        self.invalid_rgbd_step = 0

        self.agent_state = []  # Keeps track of agent state
        self.current_episode = 0  # Keeps track of current episode

        self.pseudo_goal_counter = 0  # Keeps track of how long the agent has been tracking a pseudo-goal
        self.pseudo_goal_type = 0  # Keeps track of which pseudo goal to use

        self.STUCK_PROTOCOL = False
        self.initial_agent_position = torch.tensor([self.map_size // 2, self.map_size // 2], device=self.device)

        self.collision_turn = 2

    def reset(self):
        self.pbar.update()
        self.feature_map.fill_(0.0)
        self.occupancy_map.fill_(0.0)
        self.goal_map.fill_(0.0)
        self.frontier_map.fill_(0.0)
        self.prev_actions.fill_(-1)
        self.found_call = 0
        self.total_steps = 0
        self.agent_pos = (self.map_size // 2, self.map_size // 2)  # H, W position
        self.path_found = False
        self.goal_position = torch.tensor([-1, -1], device=self.device)
        self.goals_found = 0

        self.world_projection = None
        self.binary_occupied_map.fill_(0.0)
        self.planned2Dpath = []
        self.coordinatesGrid = generate_2dgrid(self.map_size, self.map_size, False).to(
            self.device
        )
        self.waypointPose2D = None
        self.prev_agent_pos = self.agent_pos
        self.COLLISION_PROTOCOL = False
        self.collision_step = 0

        self.THRASHING_PROTOCOL = False
        self.thrashing_step = 0

        self.agent_gps_pos = torch.zeros(2)
        self.prev_agent_gps_pos = torch.zeros(2)
        self.explore_cnt = 0
        self.action_sequence = []
        self.agent_pos_sequence = []
        self.dont_go_list = []

        self.valid_obs = True
        self.INVALID_RGBD_PROTOCOL = False
        self.invalid_rgbd_step = 0

        self.agent_state = []
        self.current_episode += 1

        self.pseudo_goal_counter = 0
        self.pseudo_goal_type = 0

        self.STUCK_PROTOCOL = False
        self.initial_agent_position = torch.tensor([self.map_size // 2, self.map_size // 2], device=self.device)

        self.collision_turn = 2

    def act(self, observations):
        """
        if self.current_episode < 12:
            return {"action": 0} 
        """
        try:
            # Note that the set of possible actions (in their corresponding indices) is ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"].
            self.total_steps += 1 

            # Count number of goals found
            self.goals_found = self.count_found_goal()

            # Compute recall
            self.recall, self.precision = self.recall_and_precision()      

            for key in observations.keys():
                observations[key] = torch.tensor(observations[key], device=self.device).unsqueeze(0)
            
            self.observations = observations
            self.valid_obs = self.obs_is_valid(observations)

            self.agent_gps_pos = observations['gps'].squeeze().cpu()

            if self.total_steps > self.config.AGENT.hover_steps:
                self.found_call += 1
                self.action_sequence.append(0)
                return {"action": 0 }

            if self.prev_actions.item() == 0:
                self.prev_actions.fill_(-1)
                self.action_sequence.append(2)
                self.agent_pos_sequence.append(self.agent_pos)
                self.prev_agent_pos = self.agent_pos
                self.agent_state.append('NORMAL')
                return {"action": 2}

            # There is a weird behavior in MultiON where sometimes the first observation is completely wrong. This should be accounted for.
            if self.total_steps <= 2:
                self.agent_state.append('NORMAL')
                # Move left once and right once: a halt
                if self.total_steps == 1:
                    self.cache_pose_and_action(2)
                    self.update_agent_pose(observations)
                    return {"action": 2}
                else:
                    self.cache_pose_and_action(3)
                    self.update_agent_pose(observations)
            
                    return {"action": 3}
            elif self.total_steps <= 14:
                self.goal_map = self.localize_goal(observations)  # IMPORTANT!!!
                self.agent_state.append('NORMAL')
                self.cache_pose_and_action(3)
                self.update_agent_pose(observations)
                self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                self.binary_occupied_map[self.occupancy_map > 0.6] = 1.0  # Fill binary occupancy map

                self.occupancy_map = self.fill_dont_go(self.occupancy_map)
                self.binary_occupied_map = self.fill_dont_go(self.binary_occupied_map)

                return {"action": 3}

            elif (self.is_collision() or self.COLLISION_PROTOCOL) and ('collision' in self.config.AGENT.PLANNING.consider_tgt):
                self.agent_state.append('COLLISION')
                self.goal_map = self.localize_goal(observations)
                self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                self.binary_occupied_map[self.occupancy_map > 0.6] = 1.0  # Fill binary occupancy map

                self.occupancy_map = self.fill_dont_go(self.occupancy_map)
                self.binary_occupied_map = self.fill_dont_go(self.binary_occupied_map)

                # Dilate occupancy maps (use a temporary map or else the map will blow up)
                selem = skimage.morphology.disk(1)
                self.total_no_go_map = ((self.occupancy_map < 0.1) | (self.occupancy_map > 0.9)).float()

                self.dilated_binary_occupied_map = torch.from_numpy(skimage.morphology.binary_dilation(
                    self.binary_occupied_map.squeeze().cpu().numpy(), selem)).float().to(self.device)
                self.dilated_binary_occupied_map = self.dilated_binary_occupied_map.reshape(1, self.map_size, self.map_size, 1)
                self.dilated_occupancy_map = self.occupancy_map.clone().detach()
                self.dilated_occupancy_map[self.dilated_binary_occupied_map.nonzero(as_tuple=True)] = 1.0
                self.update_agent_pose(observations)
                if self.config.AGENT.PLANNING.collision_protocol == 'revert':
                    # If there is collision, make a 180 degree turn and take one step back
                    if self.collision_step == 6:
                        self.collision_step += 1
                        self.cache_pose_and_action(1)
                        return {"action": 1}
                    elif self.collision_step > 6 and self.collision_step < 12:
                        self.goal_position = torch.tensor([self.map_size // 2, self.map_size // 2]).long()
                        actions = self.plan_to_goal_dstar()
                        self.cache_pose_and_action(actions.item())
                        self.collision_step += 1
                        if self.collision_step == 12:
                            self.COLLISION_PROTOCOL = False
                            self.collision_step = 0
                        return {"action": actions.item()}
                    else:
                        self.COLLISION_PROTOCOL = True
                        self.collision_step += 1
                        self.cache_pose_and_action(2)
                        return {"action": 2}

                elif self.config.AGENT.PLANNING.collision_protocol == 'random':
                    if self.collision_step == self.config.AGENT.PLANNING.collision_random_steps - 1:
                        self.COLLISION_PROTOCOL = False
                        self.collision_step = 0
                        random_action = random.choice(self._POSSIBLE_ACTIONS[1:])
                        actions = torch.tensor([random_action])
                        self.cache_pose_and_action(actions.item())
                        return {"action": actions.item()}
                    
                    else:
                        self.COLLISION_PROTOCOL = True
                        self.collision_step += 1
                        random_action = random.choice(self._POSSIBLE_ACTIONS[1:])
                        actions = torch.tensor([random_action])
                        self.cache_pose_and_action(actions.item())
                        return {"action": actions.item()}
                
                elif self.config.AGENT.PLANNING.collision_protocol == 'turn':
                    # If there is collision, turn until one can make a forward action
                    if self.collision_step == 1:
                        self.cache_pose_and_action(1)
                        self.COLLISION_PROTOCOL = False
                        self.collision_step = 0
                        return {"action": 1}
                    else:
                        self.COLLISION_PROTOCOL = True
                        self.collision_step += 1

                        if self.agent_state[-2] == 'NORMAL':  # First collision since normal
                            self.collision_turn = random.choice([2, 3])
                        self.cache_pose_and_action(self.collision_turn)
                        return {"action": self.collision_turn}


                elif self.config.AGENT.PLANNING.collision_protocol == 'go_back':
                    if not self.COLLISION_PROTOCOL:  # Find rewind path
                        agent_idx = len(self.agent_state) - 1
                        valid_idx = self.get_past_state_idx(agent_idx, self.config.AGENT.PLANNING.collision_go_back_steps)
                        self.rewind_path = self.reverse_path(agent_idx - 1, valid_idx - 1)
                    self.dont_go_list.append(self.agent_pos)
                    if self.collision_step == len(self.rewind_path) - 1:  # If last
                        self.COLLISION_PROTOCOL = False
                        self.collision_step = 0
                        actions = torch.tensor([self.rewind_path[-1]])
                        self.rewind_path = []
                        self.cache_pose_and_action(actions.item())
                        return {"action": actions.item()}
                    
                    else:
                        actions = torch.tensor([self.rewind_path[self.collision_step]])
                        self.COLLISION_PROTOCOL = True
                        self.collision_step += 1
                        self.cache_pose_and_action(actions.item())
                        return {"action": actions.item()}
            else:
                self.agent_state.append('NORMAL')
                self.update_agent_pose(observations)

                self.red_goal_map = self.goal_map[..., 0: 1]
                self.green_goal_map = self.goal_map[..., 1: 2]
                self.blue_goal_map = self.goal_map[..., 2: 3]
                self.yellow_goal_map = self.goal_map[..., 3: 4]
                self.white_goal_map = self.goal_map[..., 4: 5]
                self.pink_goal_map = self.goal_map[..., 5: 6]
                self.black_goal_map = self.goal_map[..., 6: 7]
                self.cyan_goal_map = self.goal_map[..., 7: 8]
                
                with torch.no_grad():
                    self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                    self.binary_occupied_map[self.occupancy_map > 0.6] = 1.0  # Fill binary occupancy map

                    self.occupancy_map = self.fill_dont_go(self.occupancy_map)
                    self.binary_occupied_map = self.fill_dont_go(self.binary_occupied_map)

                    # Dilate occupancy maps (use a temporary map or else the map will blow up)
                    selem = skimage.morphology.disk(1)
                    self.total_no_go_map = ((self.occupancy_map < 0.1) | (self.occupancy_map > 0.9)).float()

                    self.dilated_binary_occupied_map = torch.from_numpy(skimage.morphology.binary_dilation(
                        self.binary_occupied_map.squeeze().cpu().numpy(), selem)).float().to(self.device)
                    self.dilated_binary_occupied_map = self.dilated_binary_occupied_map.reshape(1, self.map_size, self.map_size, 1)
                    self.dilated_occupancy_map = self.occupancy_map.clone().detach()
                    self.dilated_occupancy_map[self.dilated_binary_occupied_map.nonzero(as_tuple=True)] = 1.0

                    self.goal_map = self.localize_goal(observations)  # Localization of all eight goals
                    if self.check_current_goal(observations):  # Check if goal is localized in self.goal_map
                        self.goal_position = self.find_current_goal(observations)  # Find goal location and save it to goal_position
                        self.path_found = False
                        if self.reached_goal():
                            actions = torch.zeros(1).long()  # Found action
                        else:
                            actions = self.plan_to_goal_dstar()  # Planning
                    else:
                        actions = self.explore()  # Exploration

                if actions.item() == 0:
                    self.found_call += 1

                self.cache_pose_and_action(actions.item())
                return {"action": actions.item()}
        
        except:
            return {"action": 0}

    def update_agent_pose(self, observations):
        self.agent_pos = to_grid(self.map_size, self.coordinate_min, self.coordinate_max).get_grid_coords(observations['gps'])
        self.agent_pos = (self.agent_pos[0].long().item(), self.agent_pos[1].long().item())  # H, W position
        self.agent_pos_sequence.append(self.agent_pos)
        self.agent_rot = observations['compass'].item()

    def cache_pose_and_action(self, action: int):
        self.prev_actions.fill_(action)
        self.prev_agent_pos = self.agent_pos
        self.prev_agent_gps_pos = self.agent_gps_pos.clone().detach()
        self.action_sequence.append(action)

    def update_map(self, observations):
        """
        Update self.feature_map and self.occupancy_map with current observations.

        Args:
            observations: Dictionary containing observations.

        Returns:
            new_feature_map: torch.tensor of the same shape as self.feature_map containing updated feature map.
            new_occupancy_map: torch.tensor of the same shape as self.occupancy_map containing updated occupancy map.
        """
        rgb = (observations["rgb"].float() / 255.).permute(0, 3, 1, 2)
        depth = observations["depth"].clone().detach()
        compass = observations["compass"].clone().detach().fill_(0.0)  # This should not apply

        if self.valid_obs:
            # Update feature map
            ego_feature = self.feature_projection.forward(rgb, depth * 10, -(compass))  # Egocentric feature map on ground plane
            self.ego_img = ego_feature.squeeze(0).permute(1, 2, 0)
            new_feature_map = self.feature_registration.forward(observations, self.feature_map, ego_feature)  # (B, H, W, C_F)

            # Update occupancy map by adding vacant and occupied regions

            # First update vacant
            ego_vacant = torch.zeros(self.num_obs, self.occupancy_map_channels, self.egocentric_map_size, self.egocentric_map_size, device=self.device)
            for march_idx in np.arange(0, 10, self.depth_march_size):
                ego_vacant += self.occupied_projection.forward(torch.ones_like(depth, device=self.device).fill_(0.5).permute(0, 3, 1, 2), \
                    depth * march_idx, -(compass))  # Egocentric map marking vacant regions on ground plane
                ego_vacant[ego_vacant >= 0.5] = 0.5
                ego_vacant[ego_vacant < 0.5] = 0.0
            ego_vacant[:, :, self.egocentric_map_size - 1: self.egocentric_map_size + 1, self.egocentric_map_size - 1: self.egocentric_map_size + 1] = 0.5
            new_occupancy_map = self.occupancy_registration.forward(observations, self.occupancy_map, ego_vacant)  # (B, H, W, C_O)

            # Overwrite vacant with occupied regions
            ego_occupied = self.occupied_projection.forward(torch.ones_like(depth, device=self.device).permute(0, 3, 1, 2), depth * 10, \
                -(compass))  # Egocentric map marking occupied regions on ground plane
            ego_occupied[ego_occupied > 0] = 1.0
            new_occupancy_map = self.occupancy_registration.forward(observations, new_occupancy_map, ego_occupied)  # (B, H, W, C_O)
            """
            # Further overwrite vacant with floor occupied regions
            ego_floor_occupied = self.floor_occupied_projection.forward(torch.ones_like(depth, device=self.device).permute(0, 3, 1, 2), depth * 10, \
                -(compass))  # Egocentric map marking occupied regions on ground plane
            ego_floor_occupied[ego_floor_occupied > 0] = 1.0
            new_occupancy_map = self.occupancy_registration.forward(observations, new_occupancy_map, ego_floor_occupied)  # (B, H, W, C_O)
            """
            # Quantize occupancy map
            new_occupancy_map[new_occupancy_map < 0.3] = 0.0
            new_occupancy_map[(new_occupancy_map > 0.3) & (new_occupancy_map < 0.6)] = 0.5
            new_occupancy_map[new_occupancy_map > 0.6] = 1.0
        else:
            new_feature_map = self.feature_map
            new_occupancy_map = self.occupancy_map

        return new_feature_map, new_occupancy_map

    def localize_goal(self, observations):
        """
        Localize goal from self.feature_map. Here all possible objects are localized.
        Higher values in goal map indicate higher probability of being a goal.

        Args:
            observations: Dictionary containing current agent observations

        Returns:
            new_goal_map: torch.tensor of the same shape as self.goal_map containing update goal map
            with each channel representing a single object.
        """        
        # Detect goal from RGB image
        rgb = (observations["rgb"].float() / 255.)  # (B, H, W, C)
        depth = observations["depth"].clone().detach()
        compass = observations["compass"].clone().detach().fill_(0.0)  # This should not apply
        goal_image = torch.zeros(rgb.shape[0], rgb.shape[1], rgb.shape[2], 8, device=self.device)  # (B, H, W, 8)
        for idx, goal_feature in enumerate(self.goal_features):
            diff = torch.norm(rgb - goal_feature, dim=-1)  # (B, H, W)
            if idx == 4 or idx == 6:  # white, black
                threshold = 0.0001
            elif idx == 2 or idx == 1 or idx ==0 or idx == 3:  # blue, green, red, yellow
                threshold = 0.1
            else:  # cyan, pink
                threshold = 0.5

            # Quantize diff with threshold
            inlier = (diff < threshold) & (depth != 0).squeeze(-1)
            outlier = (diff > threshold) | (depth == 0).squeeze(-1)
            diff[inlier] = 1.
            diff[outlier] = 0.
            
            labels, num = skimage.measure.label(diff.squeeze(0).cpu().int().numpy(), connectivity=2, return_num=True)
            for label_idx in range(1, num + 1):  # Non-zero labels
                if idx == 4 or idx == 6:  # white, black
                    size_threshold = 50
                else:  # blue, green, red, yellow, cyan, pink
                    size_threshold = 10

                if (labels == label_idx).sum() > size_threshold:
                    inlier = torch.from_numpy((labels == label_idx)).unsqueeze(0)  # (B, H, W)
                    diff[inlier] = 1.0
                    diff[~inlier] = 0.0
                    break
                else:
                    diff.fill_(0.0)

            goal_image[..., idx] = diff

        self.goal_detect_img = goal_image
        self.red_detect_img = self.goal_detect_img[0, ..., 0: 1]
        self.green_detect_img = self.goal_detect_img[0, ..., 1: 2]
        self.blue_detect_img = self.goal_detect_img[0, ..., 2: 3]
        self.yellow_detect_img = self.goal_detect_img[0, ..., 3: 4]
        self.white_detect_img = self.goal_detect_img[0, ..., 4: 5]
        self.pink_detect_img = self.goal_detect_img[0, ..., 5: 6]
        self.black_detect_img = self.goal_detect_img[0, ..., 6: 7]
        self.cyan_detect_img = self.goal_detect_img[0, ..., 7: 8]

        goal_image = goal_image.permute(0, 3, 1, 2)
        # Detect goal from image in a binary fashion (1 if goal and 0 otherwise) and project the affinity on the map
        ego_goal = self.goal_projection.forward(goal_image, depth * 10, compass)  # Egocentric goal map on ground plane
        new_goal_map = self.goal_registration.forward(observations, self.goal_map, ego_goal)  # (B, H, W, N_obj)
        new_goal_map[new_goal_map > 0.0] = 1.0 # Make values binary

        return new_goal_map

    def check_current_goal(self, observations):
        """
        Decide if current goal is localized in self.goal_map.

        Args:
            observations: Dictionary containing observations.

        Returns:
            goal_localized: True if goal is localized.
        """
        tgt_goal_idx = observations['multiobjectgoal'].item()
        if self.goal_measure == 'threshold':
            # Check if any grid in self.goal_map is over threshold
            goal_localized = torch.any(self.goal_map[..., tgt_goal_idx] > self.goal_threshold).bool().item()
            return goal_localized

        else:
            raise ValueError("Invalid goal measure")
    
    def find_current_goal(self, observations=None, tgt_goal_idx=None):
        """
        Find the location of current goal in self.goal_map.

        Args:
            observations: Dictionary containing observations.
            tgt_goal_idx: Index of goal to look for.

        Returns:
            goal_position: torch.tensor of shape (2, ) containing goal position
        """
        assert observations is not None or tgt_goal_idx is not None

        if tgt_goal_idx is None:
            tgt_goal_idx = observations['multiobjectgoal'].item()
        
        if self.goal_measure == 'threshold':
            # Average grid locations in self.goal_map that are over threshold
            over_thres = torch.where((self.goal_map[..., tgt_goal_idx] > self.goal_threshold).squeeze(0))  # Indices for H, W
            
            goal_position = torch.tensor([over_thres[0].float().mean(), over_thres[1].float().mean()], device=self.device).long()  # H, W position
            return goal_position
        else:
            raise ValueError("Invalid goal measure")

    def plan_to_goal_dstar(self):
        self.planned2Dpath = self.plan_path()

        if self.waypointPose2D is None or self.reached_waypoint(self.waypointPose2D):
            self.waypointPose2D = self.get_valid_waypoint()

        diff_vector = torch.tensor([self.waypointPose2D[0] - self.agent_pos[0], self.waypointPose2D[1] - self.agent_pos[1]], device=self.device)
        diff_vector *= -1  # Hack to get consistent with compass
        diff_theta = torch.atan2(diff_vector[1], diff_vector[0]) % (2 * math.pi)

        tgt_theta = (diff_theta - self.agent_rot) % (2 * np.pi)
        angle_th = self.config.AGENT.PLANNING.angle_th

        # forward_ok = ((self.observations['depth'].squeeze() * 10.0) > 0.3).float().mean() > 0.8
        if (tgt_theta < angle_th or tgt_theta > 2 * np.pi - angle_th):
            actions = torch.tensor([1], device=self.device)
        elif tgt_theta < np.pi:
            actions =  torch.tensor([2], device=self.device)
        else:
            actions = torch.tensor([3], device=self.device)

        return actions
    
    def get_valid_waypoint(self):
        next_waypoint = self.planned2Dpath[0]
        agent_pos = torch.tensor([self.agent_pos[0], self.agent_pos[1]])
        while torch.norm(next_waypoint - agent_pos) < self.config.AGENT.PLANNING.min_waypoint_spacing:
            if len(self.planned2Dpath) > 1:
                self.planned2Dpath = self.planned2Dpath[1:]
                next_waypoint = self.planned2Dpath[0]
            else:
                next_waypoint = self.goal_position
                break
        
        return next_waypoint

    def prev_plan_is_not_valid(self):
        if len(self.planned2Dpath) == 0:
            return True
        pp = torch.cat(self.planned2Dpath).detach().cpu().view(-1, 2)
        binary_map = self.binary_occupied_map.squeeze().detach() >= 0.5
        obstacles_on_path = (
            binary_map[pp[:, 0].long(), pp[:, 1].long()]
        ).long().sum().item() > 0
        return obstacles_on_path  # obstacles_nearby or  obstacles_on_path

    def rawmap2_planner_ready(self, rawmap, start_map, goal_map):
        rawmap = self.preprocessNet(rawmap)
        map1 = (
            torch.clamp(rawmap, min=0, max=1.0)
            - start_map
            - F.max_pool2d(goal_map, 3, stride=1, padding=1)
        )
        return torch.relu(map1)

    def plan_path(self, overwrite=False):
        map2DObstacles = self.dilated_binary_occupied_map.permute(0, 3, 1, 2)
        self.waypointPose2D = None
        start_map = torch.zeros_like(map2DObstacles).to(self.device)
        start_map[
            0, 0, self.agent_pos[0], self.agent_pos[1]
        ] = 1.0
        goal_map = torch.zeros_like(map2DObstacles).to(self.device)
        goal_map[
            0,
            0,
            self.goal_position[0],
            self.goal_position[1],
        ] = 1.0
        path, cost = self.planner(
            self.rawmap2_planner_ready(
                map2DObstacles, start_map, goal_map
            ).to(self.device),
            self.coordinatesGrid.to(self.device),
            goal_map.to(self.device),
            start_map.to(self.device),
        )
        if len(path) == 0:
            return path
        
        map_cell_size = (self.coordinate_max - self.coordinate_min) / self.map_size
        map_size_meters = self.coordinate_max - self.coordinate_min
        return path

    def explore(self):
        """
        Make next action for exploration using self.occupancy_map, and self.feature_map.

        Args:
            None

        Returns:
            actions: (1, ) torch.tensor containing next action
        """

        explore_map = self.occupancy_map
        temp_occp = explore_map[0]
        explored_area_map_t = torch.ones(temp_occp.size())
        explored_area_map_t[temp_occp == 0.5] = 0.5 
        components_img = explored_area_map_t.reshape([self.map_size, self.map_size]).squeeze(0).cpu().float().numpy()
        components_img[self.agent_pos[0] - 2: self.agent_pos[0] + 2, self.agent_pos[1] - 2: self.agent_pos[1] + 2] = 0.5  # Make sure agent is on 'open'
        components_labels, num = skimage.morphology.label(components_img, connectivity = 2, background = 1, return_num = True)
        self.connected_idx = (components_labels == components_labels[self.agent_pos[0],self.agent_pos[1]])
        _, counts = np.unique(components_labels, return_counts=True)
        largest_area_idx = np.argmax(counts[1:]) + 1
        self.largest_open = (components_labels == largest_area_idx)

        selem = skimage.morphology.disk(1)
        map_np = temp_occp.reshape([self.map_size,self.map_size]).squeeze(0).cpu().float().numpy()
        occupied_idx = (map_np == 1)
        unexp_idx = (map_np == 0)
        empty_idx = (map_np == 0.5)
        neighbor_unexp_idx = (skimage.filters.rank.minimum((map_np * 255).astype(np.uint8), selem) == 0)
        neighbor_occp_idx = (skimage.filters.rank.maximum((map_np * 255).astype(np.uint8),selem) == 255)
        self.frontier_idx = empty_idx & neighbor_unexp_idx & (~neighbor_occp_idx)

        valid_idx = self.frontier_idx & self.connected_idx
        result_img = np.zeros([self.map_size,self.map_size,3])
        result_img[empty_idx] = [0.5, 0.5, 0.5]
        result_img[occupied_idx] = [1, 1, 1]
        result_img[unexp_idx] = [0, 0, 0]

        result_img[self.connected_idx] = [1, 0, 0]
    
        result_img[valid_idx] = [0, 1, 0]

        cluster_img = np.zeros([self.map_size, self.map_size], dtype=np.uint8)
        cluster_img[valid_idx] = 1
        labels_cluster, num = skimage.measure.label(cluster_img, connectivity = 2, return_num = True)
        if cluster_img.sum() !=0:
            unique, counts = np.unique(labels_cluster, return_counts = True)
        
            largest_cluster_label = np.argmax(counts[1:])+1
            output_img = np.zeros([self.map_size,self.map_size])
            output_img[labels_cluster == largest_cluster_label] = 1
            output_img[labels_cluster != largest_cluster_label] = 0 
            final_idx = (output_img == 1)
            result_img[final_idx] = [1,1,0]
            self.final_idx = final_idx
            x_np = np.where(final_idx == True)[0]
            y_np = np.where(final_idx == True)[1]

            x_mat = np.transpose(np.vstack([x_np]*np.size(x_np))) - x_np
            y_mat = np.transpose(np.vstack([y_np]*np.size(y_np))) - y_np
            sum_val = np.sum((x_mat**2 + y_mat**2)**(1/2),1)
            medoid = np.argmin(sum_val)

            self.goal_position = torch.tensor([x_np[medoid], y_np[medoid]], device=self.device).long() 
            result_img[x_np[medoid],y_np[medoid]] = [1,0,1]

            if not self.valid_obs:  # Block final_idx
                self.dont_go_list += torch.stack(torch.where(torch.from_numpy(final_idx)), 1).tolist()
                self.path_found = False
            else:
                self.path_found = True
        else:
            self.goal_position = None
            self.path_found = False

        dist = torch.sqrt(torch.sum((torch.tensor(self.agent_pos, device=self.device).float() - self.initial_agent_position)**2))
        if self.path_found == False and dist > 30: 
            self.goal_position = self.initial_agent_position
            self.path_found = True

        self.frontier_map[0] = torch.Tensor(result_img)
        
        if self.path_found: 
            actions = self.plan_to_goal_dstar()
            actions = torch.tensor([actions])
        else: 
            actions = np.random.choice([1,2,3])

        return actions

    def explore_mod(self):
        """
        Make next action for exploration using self.occupancy_map, and self.feature_map.

        Args:
            None

        Returns:
            actions: (1, ) torch.tensor containing next action
        """
        explore_map = self.occupancy_map
        temp_occp = explore_map[0]
        explored_area_map_t = torch.ones(temp_occp.size())
        explored_area_map_t[temp_occp == 0.5] = 0.5 
        components_img = explored_area_map_t.reshape([self.map_size, self.map_size]).squeeze(0).cpu().float().numpy()
        components_img[self.agent_pos[0] - 2: self.agent_pos[0] + 2, self.agent_pos[1] - 2: self.agent_pos[1] + 2] = 0.5  # Make sure agent is on 'open'
        components_labels, num = skimage.morphology.label(components_img, connectivity = 2, background = 1, return_num = True)
        self.connected_idx = (components_labels == components_labels[self.agent_pos[0],self.agent_pos[1]])
        _, counts = np.unique(components_labels, return_counts=True)
        largest_area_idx = np.argmax(counts[1:]) + 1
        self.largest_open = (components_labels == largest_area_idx)

        selem = skimage.morphology.square(3)
        map_np = temp_occp.reshape([self.map_size,self.map_size]).squeeze(0).cpu().float().numpy()
        occupied_idx = (map_np == 1)
        unexp_idx = (map_np == 0)
        empty_idx = (map_np == 0.5)
        neighbor_unexp_idx = (skimage.filters.rank.minimum((map_np * 255).astype(np.uint8), selem) == 0)
        neighbor_occp_idx = (skimage.filters.rank.maximum((map_np * 255).astype(np.uint8),selem) == 255)
        self.frontier_idx = empty_idx & neighbor_unexp_idx & (~neighbor_occp_idx)

        valid_idx = self.frontier_idx & self.connected_idx
        result_img = np.zeros([self.map_size,self.map_size,3])
        result_img[empty_idx] = [0.5, 0.5, 0.5]
        result_img[occupied_idx] = [1, 1, 1]
        result_img[unexp_idx] = [0, 0, 0]

        result_img[self.connected_idx] = [1, 0, 0]
    
        result_img[valid_idx] = [0, 1, 0]

        cluster_img = np.zeros([self.map_size, self.map_size], dtype=np.uint8)
        cluster_img[valid_idx] = 1
        labels_cluster, num = skimage.measure.label(cluster_img, connectivity = 2, return_num = True)
        if cluster_img.sum() !=0:
            unique, counts = np.unique(labels_cluster, return_counts = True)
            goal_position_list = []
            label_list = []
            final_idx_list = []
            for i in unique[counts >= 3][1:]:
                output_img = np.zeros([self.map_size, self.map_size])
                output_img[labels_cluster == i] = 1
                output_img[labels_cluster != i] = 0
                final_idx = (output_img == 1)
                x_np = np.where(final_idx == True)[0]
                y_np = np.where(final_idx == True)[1]
                x_mat = np.transpose(np.vstack([x_np]*np.size(x_np))) - x_np
                y_mat = np.transpose(np.vstack([y_np]*np.size(y_np))) - y_np
                sum_val = np.sum((x_mat**2 + y_mat**2)**(1/2),1)
                medoid_idx = np.argmin(sum_val)
                medoid = [x_np[medoid_idx], y_np[medoid_idx]]
                goal_position_list.append(medoid)
                label_list.append(i)
                final_idx_list.append(final_idx)
            
            if goal_position_list != []:
                self.path_found = True
                agent_pos = np.array(self.agent_pos, dtype=np.float)
                farthest_idx = np.argmin(np.sqrt(np.sum((goal_position_list - agent_pos)**2,1)))
                self.goal_position = torch.tensor(goal_position_list[farthest_idx], device=self.device).long()
                self.final_idx = final_idx_list[farthest_idx]
                result_img[goal_position_list[farthest_idx][0], goal_position_list[farthest_idx][1]] = [1,0,1]
            
            if not self.valid_obs:  # Block final_idx
                self.dont_go_list += torch.stack(torch.where(torch.from_numpy(self.final_idx)), 1).tolist()
                self.path_found = False


        dist = torch.sqrt(torch.sum((torch.tensor(self.agent_pos, device=self.device).float() - self.initial_agent_position)**2))
        if self.path_found == False and dist > 30: 
            self.goal_position = self.initial_agent_position
            self.path_found = True

        self.frontier_map[0] = torch.Tensor(result_img)

        if self.path_found:         
            actions = self.plan_to_goal_dstar()
            actions = torch.tensor([actions])
        else: 
            actions = np.random.choice([1,2,3])


        return actions

    def fill_features(self):
        """
        Fill self.goal_features with goal-related prior information.

        Args:
            None

        Returns:
            goal_features: (N_goal, N_features) tensor containing goal feature information
        """
        return torch.tensor([
            [1., 0., 0.0017],  # red
            [0., 0.1897, 0.], # green
            [0.0018, 0.0037, 0.5288],  # blue
	        [1., 0.9310, 0.],  # yellow
            [1., 1., 1.],  # white
            [0.969, 0.0001, 1.],  # pink
            [0., 0., 0.],  # black
            [0., 1., 1.]  # cyan
        ], device=self.device)

    def fill_goal_dict(self):
        """
        Fill self.goal_dict with goal-related prior information.

        Args:
            None

        Returns:
            goal_dict: Dictionary containing mapping from goal name to goal index
        """

        return {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2,
            'cylinder_yellow':3, 'cylinder_white':4, 'cylinder_pink':5, 'cylinder_black':6, 'cylinder_cyan':7
        }

    def reached_goal(self):
        """
        Determine if the agent has reached the goal.

        Args:
            None

        Returns:
            goal_found: Boolean indicating whether goal is reached by agent
        """
        grid_size = (self.coordinate_max - self.coordinate_min) / self.map_size
        grid_thres = 0.7 / grid_size
        goal_found = torch.norm(self.goal_position.float() - torch.tensor([self.agent_pos[0], self.agent_pos[1]], device=self.device).float()) < grid_thres
        if goal_found:
            self.goal_position = None  # Reset
            return True
        else:
            return False

    def reached_waypoint(self, waypoint):
        """
        Determine if the agent has reached the waypoint.

        Args:
            waypoint: (2, ) tensor containing waypoint indices

        Returns:
            reached: Boolean indicating whether waypoint is reached by agent
        """
        grid_size = (self.coordinate_max - self.coordinate_min) / self.map_size
        grid_thres = round(self.config.AGENT.PLANNING.waypoint_thres / grid_size)
        reached = torch.norm(waypoint.float() - torch.tensor([self.agent_pos[0], self.agent_pos[1]]).float()) < grid_thres
        if reached:
            return True
        else:
            return False
    
    def count_found_goal(self):
        """
        Count number of goals found.

        Args:
            None
        
        Returns:
            goal_count: Integer containing number of goals found
        """
        goal_count = (self.goal_map.flatten(0, 2).sum(0) > 0).sum().item()

        return goal_count
    

    def recall_and_precision(self):
        """
        Compute recall and precision of the target goals.

        Args:
            None
        
        Returns:
            goal_recall, goal_precision: Float containing recall and precision of objects found
        """
        goals = [self.goal_dict[multinav_goal.object_category] for multinav_goal in self.challenge._env.current_episode.goals]
        goal_found_tensor = (self.goal_map.flatten(0, 2).sum(0) > 0).tolist()

        goal_recall = sum([goal_found_tensor[idx] for idx in goals]) / len(self.challenge._env.current_episode.goals)
        if sum(goal_found_tensor) != 0:
            goal_precision = sum([goal_found_tensor[idx] for idx in goals]) / sum(goal_found_tensor)
        else:
            goal_precision = 0.0

        return goal_recall, goal_precision

    def distance_recall_and_precision(self, dist_threshold=1.5):
        """
        Compute recall and precision of the target goals, taking the distance to goal into account.

        Args:
            dist_threshold: Threshold value to call goal predictions correct
        
        Returns:
            goal_distance_recall, goal_distance_precision: Float containing recall and precision of objects found taking distance into account
        """
        goals = [self.goal_dict[multinav_goal.object_category] for multinav_goal in self.challenge._env.current_episode.goals]
        gt_world_goal_location = [torch.tensor(goal.position, device=self.device) for goal in self.challenge._env.current_episode.goals]
        goal_found_tensor = (self.goal_map.flatten(0, 2).sum(0) > 0).tolist()
        valid_found = 0

        for goal_idx in goals:
            goal_position = self.find_current_goal(tgt_goal_idx=goal_idx)  # H, W coordinates
            world_goal_position = self.world_projection.convert(goal_position)  # (3, ) tensor

            if torch.norm(world_goal_position - gt_world_goal_location) < dist_threshold:
                valid_found += 1

        goal_distance_recall = valid_found / len(self.challenge._env.current_episode.goals)
        if sum(goal_found_tensor) != 0:
            goal_distance_precision = valid_found / sum(goal_found_tensor)
        else:
            goal_distance_precision = 0.0

        return goal_distance_recall, goal_distance_precision

    def is_collision(self):
        dist = torch.norm(self.agent_gps_pos - self.prev_agent_gps_pos)
        if self.prev_actions.item() == 1 and dist < self.config.AGENT.PLANNING.collision_thres:
            # Fill collided regions in occupancy map
            if self.config.AGENT.PLANNING.collision_fill_in == 'current_only':
                self.dont_go_list.append(self.agent_pos)
            
            elif self.config.AGENT.PLANNING.collision_fill_in == 'current_and_past':
                self.dont_go_list.append(self.agent_pos)
                search_idx = self.get_past_position_idx(len(self.action_sequence) - 1, 1)
                self.dont_go_list.append(self.agent_pos_sequence[search_idx])

            elif self.config.AGENT.PLANNING.collision_fill_in == 'linear':
                # Fill all rasterized line
                self.dont_go_list.append(self.agent_pos)
                # Backtrack position history
                search_idx = self.get_past_position_idx(len(self.action_sequence) - 1, 1)
                self.dont_go_list.append(self.agent_pos_sequence[search_idx])
                i_idx_prev, j_idx_prev = self.agent_pos_sequence[search_idx]
                i_idxs = np.linspace(start=i_idx_prev, stop=self.agent_pos[0], num=4)
                j_idxs = np.linspace(start=j_idx_prev, stop=self.agent_pos[1], num=4)
                for i_idx, j_idx in zip(i_idxs, j_idxs):
                    self.dont_go_list.append((int(i_idx), int(j_idx)))

            elif self.config.AGENT.PLANNING.collision_fill_in == 'padding':
                self.dont_go_list.append((self.agent_pos[0] - 1, self.agent_pos[1] - 1))
                self.dont_go_list.append((self.agent_pos[0] - 1, self.agent_pos[1]))
                self.dont_go_list.append((self.agent_pos[0] - 1, self.agent_pos[1] + 1))
                self.dont_go_list.append((self.agent_pos[0], self.agent_pos[1] - 1))
                self.dont_go_list.append((self.agent_pos[0], self.agent_pos[1]))
                self.dont_go_list.append((self.agent_pos[0], self.agent_pos[1] + 1))
                self.dont_go_list.append((self.agent_pos[0] + 1, self.agent_pos[1] - 1))
                self.dont_go_list.append((self.agent_pos[0] + 1, self.agent_pos[1]))
                self.dont_go_list.append((self.agent_pos[0] + 1, self.agent_pos[1] + 1))

            else:
                pass

            return True
        return False
    
    def visualize_path(self, path):
        # path is a list containing tensors of shape (2, ) where each tensor indicates a H, W coordinate of waypoint
        vis_map = torch.cat([self.occupancy_map.clone().detach()] * 3, -1)
        for waypoint in path:
            vis_map[:, waypoint[0].long() - 1: waypoint[0].long() + 1, waypoint[1].long() - 1: waypoint[1].long() + 1, :] = torch.tensor([1., 0., 0.,])

        return vis_map
    
    def reverse_path(self, start_idx, end_idx):
        """
        Reverse path starting from start_idx to end_idx

        Args:
            start_idx: Starting index
            end_idx: Ending index

        Return:
            action_list: List of reversed actions
        """
        if not(start_idx >= 0 and end_idx >= 0):
            # Do revert
            return [2, 2, 2, 2, 2, 2, 1]

        action_list = []
        action_idx = 0
        if start_idx == end_idx:
            return action_list
        elif start_idx > end_idx:
            step = 1
        else:
            step = -1

        seen_forward = False
        while start_idx - action_idx >= end_idx:
            if self.action_sequence[start_idx - action_idx] == 1:
                if seen_forward:
                    action_list.append(1)
                else:
                    seen_forward = True
                    action_list += [2, 2, 2, 2, 2, 2, 1]  # 180 degree turn + forward
            elif self.action_sequence[start_idx - action_idx] == 2:
                action_list.append(3)
            elif self.action_sequence[start_idx - action_idx] == 3:
                action_list.append(2)
            action_idx += step
        """ 
        if seen_forward:  # Add another 180 degree turn to make resulting pose identical
            action_list += [2, 2, 2, 2, 2, 2]
        """
        return action_list
    
    def get_past_position_idx(self, curr_idx, k):
        # Get index of k th past position from curr_idx
        assert curr_idx >= 0
        search_idx = curr_idx
        agent_pos = self.agent_pos_sequence[curr_idx]
        num_unique = 0

        while search_idx >= 0:
            if self.agent_pos_sequence[search_idx] != agent_pos:
                num_unique += 1
                if num_unique == k:
                    break
            search_idx -= 1
        
        return search_idx

    def get_past_state_idx(self, curr_idx, k, state='NORMAL', return_if_none=-1, exclude_self=False):
        # Get index of k th past state idx from curr_idx
        assert curr_idx >= 0
        if exclude_self:
            if curr_idx >= 1:
                search_idx = curr_idx - 1
            else:
                return 0

        search_idx = curr_idx
        num_unique = 0

        while search_idx >= 0:
            if self.agent_state[search_idx] == state:
                num_unique += 1
                if num_unique == k:
                    break
            search_idx -= 1
        
        if search_idx >= 0:
            return search_idx
        else:
            return return_if_none

    def fill_dont_go(self, tgt_map):
        assert len(tgt_map.shape) == 4
        for pos in self.dont_go_list:
            tgt_map[:, pos[0], pos[1], :] = 1.0
        
        return tgt_map

    def is_thrashing(self):
        action_lens = [5, 6, 7, 8, 9, 10]
        if len(self.action_sequence) < 30:
            return False

        for action_len in action_lens:
            if self.action_sequence[-action_len:] == self.action_sequence[- 2 * action_len: -action_len]:
                if not all([action == 1 for action in self.action_sequence[-action_len:]]):
                    return True
        return False
    
    def is_stuck(self):
        pos_lens = list(range(5, 70))
        if len(self.agent_pos_sequence) < 30:
            return False

        for pos_len in pos_lens:
            if self.agent_pos_sequence[-pos_len:] == self.agent_pos_sequence[- 2 * pos_len: -pos_len]:
                return True
        return False

    def obs_is_valid(self, observations):
        rgb = (observations["rgb"].float() / 255.).permute(0, 3, 1, 2)
        rgb = rgb[..., rgb.shape[-2] // 2:, :]
        if (rgb.squeeze(0).sum(0) == 0).float().mean() > self.config.AGENT.PLANNING.hole_thres:  # If majority is zero pixels, it is probably a hole
            return False
        else:
            return True
    
    def set_pseudo_goal(self):
        if self.pseudo_goal_counter == 0:
            """
            if self.pseudo_goal_type == 0:
                self.pseudo_goal_position = (self.map_size - 1, self.map_size - 1)  # Go to opposite!
            elif self.pseudo_goal_type == 1:
                self.pseudo_goal_position = (self.map_size - 1, 0)  # Go to opposite!
            elif self.pseudo_goal_type == 2:
                self.pseudo_goal_position = (0, self.map_size - 1)  # Go to opposite!
            else:
                self.pseudo_goal_position = (0, 0)  # Go to opposite!
            """
            valid_idx = self.largest_open
            valid_points = torch.stack(torch.where(torch.from_numpy(valid_idx)), 1).float()
            max_idx = (valid_points - torch.tensor(self.agent_pos).unsqueeze(0)).norm(dim=-1).argmax()
            self.pseudo_goal_position = (valid_points[max_idx].long()[0].item(), valid_points[max_idx].long()[1].item())

            self.pseudo_goal_counter += 1
        elif self.pseudo_goal_counter == self.config.AGENT.PLANNING.max_pseudo_goal_step:
            self.pseudo_goal_counter = 0
            """
            self.pseudo_goal_type += 1
            self.pseudo_goal_type %= 4
            """
        else:
            self.pseudo_goal_counter += 1
        
        self.goal_position = self.pseudo_goal_position

def get_agent(exp_config, challenge, checkpoint_path=None):
    return DStarSGoLAMWalker(exp_config, challenge, checkpoint_path)
