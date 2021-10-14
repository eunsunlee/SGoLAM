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

class ProtoSGoLAMWalker(habitat.Agent):
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

        self.waypoints = []
        self.local_waypoints = []

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
            total_episodes = challenge.num_episodes
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
            -1.0, -0.8, 'max')
        self.occupied_projection = Projection(self.egocentric_map_size, self.map_size, self.device, self.coordinate_min, self.coordinate_max, \
            -0.6, 0.0, 'max')
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
        self.prev_pos = 0 
        self.path_found = False


        self.local_path_found = False 
        self.local_prev_pos = 0 

        

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
        self.waypoints = []
        self.prev_pos = 0 
        self.path_found = False
        self.goal_position = torch.tensor([-1, -1], device=self.device)
        self.goals_found = 0

        self.local_waypoints = []
        self.local_path_found = False 
        self.local_prev_pos = 0

        self.world_projection = None
    

    def act(self, observations):
        # Note that the set of possible actions (in their corresponding indices) is ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"].
        
        # Count number of goals found
        self.goals_found = self.count_found_goal()

        # Compute recall
        self.recall, self.precision = self.recall_and_precision()      

        for key in observations.keys():
            observations[key] = torch.tensor(observations[key], device=self.device).unsqueeze(0)
        
        if self.prev_actions.item() == 0:
            print("previous action was 0 so return 2 ")
            self.prev_actions.fill_(-1)            
            return {"action": 2}

        # There is a weird behavior in MultiON where sometimes the first observation is completely wrong. This should be accounted for.
        if self.total_steps < 2:
            # Move left once and right once: a halt
            if self.total_steps == 0:
                self.total_steps += 1
                return {"action": 2}
            else:
                self.total_steps += 1
                return {"action": 3}
        else:
            self.agent_pos = to_grid(self.map_size, self.coordinate_min, self.coordinate_max).get_grid_coords(observations['gps'])
            self.agent_pos = (self.agent_pos[0].long().item(), self.agent_pos[1].long().item())  # H, W position
            self.agent_rot = observations['compass'].item()

            self.red_goal_map = self.goal_map[..., 0: 1]
            self.green_goal_map = self.goal_map[..., 1: 2]
            self.blue_goal_map = self.goal_map[..., 2: 3]
            self.yellow_goal_map = self.goal_map[..., 3: 4]
            self.white_goal_map = self.goal_map[..., 4: 5]
            self.pink_goal_map = self.goal_map[..., 5: 6]
            self.black_goal_map = self.goal_map[..., 6: 7]
            self.cyan_goal_map = self.goal_map[..., 7: 8]
            
            if self.config.AGENT.mode == 'debug_default':
                self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                self.goal_map = self.localize_goal(observations)  # Localization of all eight goals
                self.total_steps += 1

                if self.total_steps >= self.config.AGENT.hover_steps:
                    return {"action": 0}
                else:
                    return {"action": np.random.choice([1, 2, 3])}
            
            elif self.config.AGENT.mode == 'debug_world_coord':
                if self.total_steps == 2:
                    self.world_projection = GlobalMap2World(orig_position=observations['agent_position'],
                        orig_rot=observations['heading'], grid_mapper=to_grid(self.map_size, self.coordinate_min, self.coordinate_max))

                self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                self.goal_map = self.localize_goal(observations)  # Localization of all eight goals
                gt_world_goal_location = [torch.tensor(goal.position, device=self.device) for goal in self.challenge._env.current_episode.goals]
                gt_idx = [self.world_projection.inv_convert(goal) for goal in gt_world_goal_location]
                self.feature_map[0, gt_idx[0][0] - 2: gt_idx[0][0] + 2, gt_idx[0][1] - 2: gt_idx[0][1] + 2] = torch.tensor([[1., 0., 0.]])
                self.feature_map[0, gt_idx[1][0] - 2: gt_idx[1][0] + 2, gt_idx[1][1] - 2: gt_idx[1][1] + 2] = torch.tensor([[0., 1., 0.]])
                self.feature_map[0, gt_idx[2][0] - 2: gt_idx[2][0] + 2, gt_idx[2][1] - 2: gt_idx[2][1] + 2] = torch.tensor([[0., 0., 1.]])
                self.total_steps += 1

                if self.total_steps >= self.config.AGENT.hover_steps:
                    return {"action": 0}
                else:
                    return {"action": np.random.choice([1, 2, 3])}            
            
            elif self.config.AGENT.mode == 'debug_explore':

                with torch.no_grad():
                    self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                    self.goal_map = self.localize_goal(observations)  # Localization of all eight goals

                    actions = self.explore()  # Exploration
       
                self.total_steps += 1 

                if self.total_steps >= self.config.AGENT.hover_steps:
                    return {"action": 0 }
                if actions.item() == 0:
                    self.found_call += 1
                if self.found_call == self.N_ON:
                    self.challenge._env._task.is_stop_called = True

                return {"action": actions.item()}

            elif self.config.AGENT.mode == 'debug_astar_planning':

                with torch.no_grad():
                    self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                    self.goal_map = self.localize_goal(observations)  # Localization of all eight goals
                    if self.check_current_goal(observations):  # Check if goal is localized in self.goal_map
                        self.goal_position = self.find_current_goal(observations)  # Find goal location and save it to goal_position
                        if self.reached_goal():
                            actions = torch.zeros(1).long()  # Found action
                        else:
                            actions = self.plan_to_goal_astar()  # Planning
                    else:
                        actions = self.explore()  # Exploration


                self.total_steps += 1 

                if self.total_steps >= self.config.AGENT.hover_steps:
                    return {"action": 0 }

                if actions.item() == 0:
                    self.found_call += 1
                if self.found_call == self.N_ON:
                    self.challenge._env._task.is_stop_called = True

                self.prev_actions.fill_(actions.item())
                return {"action": actions.item()}                  
            
            elif self.config.AGENT.mode == 'debug_planning':
                # TODO: @mingi
                self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                self.goal_map = self.localize_goal(observations)  # Localization of all eight goals

                self.total_steps += 1
                print(f"STEPS : {self.total_steps} GOALS_FOUND : {self.goals_found} GOALS : {[self.goal_dict[multinav_goal.object_category] for multinav_goal in self.challenge._env.current_episode.goals]}")
                if self.check_current_goal(observations) :  # Check if goal is localized in self.goal_map
                    self.goal_position = self.find_current_goal(observations)  # Find goal location and save it to goal_position
                    if self.reached_goal():
                        print("FOUND")
                        return {"action": 0}  # Found action
                    else:
                        print("PLAN")
                        return {"actions": self.plan_to_goal().item()}
                else:
                    print("EXPLORE")
                    action = self.explore()
                    return {"action": action.item()}
            
            elif self.config.AGENT.mode == 'sgolam':
                with torch.no_grad():
                    self.feature_map, self.occupancy_map = self.update_map(observations)  # Map update with observations
                    self.goal_map = self.localize_goal(observations)  # Localization of all eight goals

                    if self.check_current_goal(observations):  # Check if goal is localized in self.goal_map
                        self.goal_position = self.find_current_goal(observations)  # Find goal location and save it to goal_position
                        if self.reached_goal():
                            actions = torch.zeros(1) # Found action
                        else:
                            actions = self.plan_to_goal(observations)  # Planning
                    else:
                        actions = self.explore()  # Exploration

                if actions.item() == 0:
                    self.found_call += 1
                if self.found_call == self.N_ON:
                    self.challenge._env._task.is_stop_called = True

                self.prev_actions.fill_(actions.item())

                return {"action": actions.item()}


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

        new_occupancy_map = self.occupancy_registration.forward(observations, self.occupancy_map, ego_vacant)  # (B, H, W, C_O)

        # Overwrite vacant with occupied regions
        ego_occupied = self.occupied_projection.forward(torch.ones_like(depth, device=self.device).permute(0, 3, 1, 2), depth * 10, \
            -(compass))  # Egocentric map marking occupied regions on ground plane
        ego_occupied[ego_occupied > 0] = 1.0
        new_occupancy_map = self.occupancy_registration.forward(observations, new_occupancy_map, ego_occupied)  # (B, H, W, C_O)

        # Quantize occupancy map
        new_occupancy_map[new_occupancy_map < 0.3] = 0.0
        new_occupancy_map[(new_occupancy_map > 0.3) & (new_occupancy_map < 0.9)] = 0.5
        new_occupancy_map[new_occupancy_map > 0.9] = 1.0 

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

        if self.affinity_measure == 'conv':
            # Find goal-wise affinity by performing convolution with the goal features
            feature_map = self.feature_map.permute([0, 3, 1, 2])  # (B, C, H, W)
            new_goal_map = nn.functional.conv2d(feature_map, weight=self.goal_features.reshape(self.total_objects, self.feature_map_channels, 1, 1))
            new_goal_map = new_goal_map.permute([0, 2, 3, 1])

        elif self.affinity_measure == 'l2_diff':
            # Find goal-wise affinity by computing differences with the goal features
            new_goal_map = torch.zeros_like(self.goal_map)
            for idx, goal_feature in enumerate(self.goal_features):
                new_goal_map[..., idx] = math.sqrt(3.) - torch.norm(self.feature_map - goal_feature, dim=-1)
                new_goal_map[self.feature_map.sum(-1) < 0.01] = 0.
        
        elif self.affinity_measure == 'equality_margin':
            # Find goal-wise affinity by checking whether each feature is within the boundary of the goal feature
            new_goal_map = torch.zeros_like(self.goal_map)
            for idx, goal_feature in enumerate(self.goal_features):
                diff = torch.norm(self.feature_map - goal_feature, dim=-1)
                if idx == 4:  # white
                    threshold = 0.0001
                elif idx == 6:  # black
                    threshold = 0.00001
                elif idx == 2 or idx == 1 or idx ==0 or idx == 3:  # blue, green, red, yellow
                    threshold = 0.1
                else:  # cyan, pink, red
                    threshold = 0.5
                
                inlier = diff < threshold
                outlier = diff > threshold
                diff[inlier] = 1.
                diff[outlier] = 0.
                new_goal_map[..., idx] = diff
                new_goal_map[self.feature_map.sum(-1) < 0.01] = 0.
        
        elif self.affinity_measure == 'binary_image_detect':
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
                else:  # cyan, pink, red
                    threshold = 0.5

                # Quantize diff with threshold
                inlier = (diff < threshold) & (depth != 0).squeeze(-1)
                outlier = (diff > threshold) | (depth == 0).squeeze(-1)
                diff[inlier] = 1.
                diff[outlier] = 0.
                
                size_threshold = 50
                labels, num = skimage.measure.label(diff.squeeze(0).cpu().int().numpy(), connectivity=2, return_num=True)
                for label_idx in range(1, num + 1):  # Non-zero labels
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

        elif self.affinity_measure == 'hsv_metric':
            assert self.feature_map_channels == 3, \
            'feature_map_channels should be 3, not {}'.format(
                self.feature_map_channels
            )
            new_goal_map = []
            for batch_idx in range(self.feature_map.shape[0]):
                feature_img = self.feature_map[batch_idx].cpu().numpy()
                new_goal_map_single = []
                for obs_idx in range(self.num_obs):
                    # tensor of map_size x map_size x feature_map_channles
                    color = self.goal_features[obs_idx].cpu().numpy()
                    hsv_dist = compute_hsv_metric_from_rgb(feature_img, color)
                    hsv_dist = torch.tensor(hsv_dist)
                    new_goal_map_single.append((hsv_dist < 0.1).float().to(self.device))
                new_goal_map.append(torch.stack(new_goal_map_single, dim=2))
            new_goal_map = torch.stack(new_goal_map, dim=0)
        else:
            raise ValueError("Invalid affinity measure")

        return new_goal_map

    def plan_to_goal(self):
        """
        Make next action for exploitation using self.goal_map, self.occupancy_map, and self.feature_map.

        Args:
            None

        Returns:
            actions: (1, ) torch.tensor containing next action
        """
        
        #####  PLANNING OPTIONS  #####
        close_small_openings = False
        num_erosions = 2
        ##############################
        
        forward_step_size = self.forward_step_size
        state = [self.agent_pos[0], self.agent_pos[1], self.agent_rot]
        goal = [self.goal_position[0], self.goal_position[1]]

        not_traversible = self.occupancy_map[0,:,:,0].clone().detach().cpu() < 0.1
        not_traversible = not_traversible > 0.9
        
        if close_small_openings:
            selem = skimage.morphology.disk(2)
            selem_small = skimage.morphology.disk(1)
            traversible = skimage.morphology.binary_dilation(not_traversible, selem) != True
            n = num_erosions
            reachable = False
            while n >= 0 and not reachable:
                traversible_open = traversible.copy()
                for i in range(n):
                    traversible_open = skimage.morphology.binary_erosion(traversible_open, selem_small)
                for i in range(n):
                    traversible_open = skimage.morphology.binary_dilation(traversible_open, selem_small)
                planner = FMMPlanner(traversible_open, forward_step_size, 360//30)
                reachable = planner.set_goal(goal)
                reachable = reachable[int(round(state[1])), int(round(state[0]))]
                n = n-1
        else:
            traversible = not_traversible != True
            planner = FMMPlanner(traversible,forward_step_size, 360//30)
            reachable = planner.set_goal(goal)
        
        actions, state, act_seq = planner.get_action(state)
        for i in range(len(act_seq)):
            if act_seq[i] == 3:
                act_seq[i] = 0
            elif act_seq[i] == 0:
                act_seq[i] = 3
        if actions == 3:  # TODO: What does this mean? Why are actions swapped?
            actions = 0
        elif actions == 0:
            actions = 3
        # TODO: (Suggestion) Make act_seq as a class attribute, i.e., self.act_seq?
        # TODO: Change 'a' to 'actions' (naming)
        return torch.tensor([actions]), act_seq #TODO : return as tensor (made temporary fix)
     
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

    def plan_to_goal_astar(self):
        actions = 5
        
        end = (0, self.goal_position[0], self.goal_position[1], 1)
        
        considered_cells, self.local_waypoints, self.local_path_found = astar_algorithm(self.occupancy_map, self.agent_pos, end, self.num_obs, self.map_size, self.occupancy_map_channels , self.device)
        if self.local_path_found : 
            for cell in self.local_waypoints: 
                (a,i,j,b) = cell
                self.frontier_map[0,i,j,0] = 0
                self.frontier_map[0,i,j,1] = 1
                self.frontier_map[0,i,j,2] = 0   
        
            actions, self.local_prev_pos, self.local_waypoints, self.local_path_found, collision = get_action(self.agent_pos, self.local_waypoints, self.agent_rot, self.local_prev_pos, self.local_path_found)
           
        if actions == 5: 
            actions = np.random.choice([1,2,3])
        actions = torch.tensor([actions])
        return actions

    def explore(self):
        """
        Make next action for exploration using self.occupancy_map, and self.feature_map.

        Args:
            None

        Returns:
            actions: (1, ) torch.tensor containing next action
        """
        actions = 5
       
        frontier = Frontier(self.map_size, self.occupancy_map_channels, device, self.num_obs, self.occupancy_map)

        if (self.total_steps  % 50 == 0 and not self.path_found):
            clusters = []
            centroid = []
            centroidFound = False
        
            frontier.calculate_clusters(clusters)


            if (len(clusters)!= 0):
                for cluster in clusters:
                    centroid = frontier.calculate_centroid(cluster)
                    #frontier_temp.expandObstacles()
                   
                    considered_cells, self.waypoints, self.path_found = astar_algorithm(self.occupancy_map, self.agent_pos, centroid, self.num_obs, self.map_size, self.occupancy_map_channels , self.device)
                            
                    if self.path_found:
                        break

 
        actions, pre_pos, self.waypoints, self.path_found, collision = get_action(self.agent_pos, self.waypoints, self.agent_rot, self.prev_pos, self.path_found)

        self.prev_pos = pre_pos

        if actions == 5: 
            actions = np.random.choice([1,2,3])
        actions = torch.tensor([actions])


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
        grid_thres = 1.0 / grid_size
        goal_found = torch.norm(self.goal_position.float() - torch.tensor([self.agent_pos[0], self.agent_pos[1]], device=self.device).float()) < grid_thres
        if goal_found:
            self.goal_position = None  # Reset
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

def get_agent(exp_config, challenge, checkpoint_path=None):
    return ProtoSGoLAMWalker(exp_config, challenge, checkpoint_path)
