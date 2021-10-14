from einops import rearrange
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

"""
Code adapted from https://github.com/saimwani/multiON
"""

def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


class to_grid():
    def __init__(self, global_map_size, coordinate_min, coordinate_max):
        self.global_map_size = global_map_size
        self.coordinate_min = coordinate_min
        self.coordinate_max = coordinate_max
        self.grid_size = (coordinate_max - coordinate_min) / global_map_size

    def get_grid_coords(self, positions):
        grid_x = ((self.coordinate_max - positions[:, 0]) / self.grid_size).floor()
        grid_y = ((positions[:, 1] - self.coordinate_min) / self.grid_size).floor()
        return grid_x, grid_y
    
    def get_gps_coords(self, idx):
        # H, W indices to gps coordinates
        grid_x = idx[0].item()
        grid_y = idx[1].item()

        gps_x = self.coordinate_max - grid_x * self.grid_size
        gps_y = self.coordinate_min + grid_y * self.grid_size

        return gps_x, gps_y


class ComputeSpatialLocs():
    def __init__(self, egocentric_map_size, global_map_size, 
        device, coordinate_min, coordinate_max, height_min, height_max
    ):
        # Note: The side of one grid in egocentric map and global map is the same

        self.device = device
        self.cx, self.cy = 256./2., 256./2.     # Hard coded camera parameters
        self.fx = self.fy =  (256. / 2.) / np.tan(np.deg2rad(79 / 2.))
        self.egocentric_map_size = egocentric_map_size
        self.local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
        self.height_min, self.height_max = height_min, height_max  # Minimum, maximum height values to cut off for mapping

    def compute_y(self, depth):
        # Returns a y-value image with the same shape as depth
        depth = depth.permute(0, 3, 1, 2)
        _, _, imh, imw = depth.shape   # batchsize, 1, imh, imw
        y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(self.device)
        yy   = (y - self.cy) / self.fy
        Z = depth
        Y = yy * Z
        Y = Y.permute(0, 2, 3, 1)

        return Y[0].expand(Y.shape[1], Y.shape[2], 3)
    
    def draw_range(self, depth, tgt_img, height_min, height_max):
        # Returns a image with the height ranges in [height_min, height_max] marked in red
        tgt_img = tgt_img.clone().detach()
        depth = depth.permute(0, 3, 1, 2)
        _, _, imh, imw = depth.shape   # batchsize, 1, imh, imw
        y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(self.device)
        yy   = (y - self.cy) / self.fy
        Z = depth
        Y = yy * Z
        Y = Y.permute(0, 2, 3, 1)
        _, idx_h, idx_w, _ = torch.where((Y < height_max) & (Y > height_min))
        tgt_img[0, idx_h, idx_w, :] = torch.ByteTensor([[255, 0, 0]]).to(self.device)

        return tgt_img[0]

    def forward(self, depth):
        depth = depth.permute(0, 3, 1, 2)
        _, _, imh, imw = depth.shape   # batchsize, 1, imh, imw
        x    = rearrange(torch.arange(0, imw), 'w -> () () () w').to(self.device)
        y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(self.device)
        xx   = (x - self.cx) / self.fx
        yy   = (y - self.cy) / self.fy
        
        # 3D real-world coordinates (in meters)
        Z = depth
        X = xx * Z
        Y = yy * Z

        # Valid inputs (depth sensor's max depth is 10m)
        valid_inputs = (depth != 0)  & ((Y > self.height_min) & (Y < self.height_max)) & (depth < 10.0)

        # X ground projection and Y ground projection
        x_gp = ( (X / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, imh, imw, 1)
        y_gp = (-(Z / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, imh, imw, 1)

        return torch.cat([x_gp, y_gp], dim=1), valid_inputs


class ProjectToGroundPlane():
    def __init__(self, egocentric_map_size, device, scatter_mode):
        self.egocentric_map_size = egocentric_map_size
        self.device = device
        self.scatter_mode = scatter_mode

    def forward(self, img, spatial_locs, valid_inputs):
        outh, outw = (self.egocentric_map_size, self.egocentric_map_size)
        bs, f, HbyK, WbyK = img.shape
        K = 1
        # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
        idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(self.device), \
                    (torch.arange(0, WbyK, 1)*K).long().to(self.device))

        spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
        valid_inputs_ss = valid_inputs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
        valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
        invalid_inputs_ss = ~valid_inputs_ss

        # Filter out invalid spatial locations
        invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                            (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

        invalid_writes = invalid_spatial_locs | invalid_inputs_ss

        # Set the idxes for all invalid locations to (0, 0)
        spatial_locs_ss[:, 0][invalid_writes] = 0
        spatial_locs_ss[:, 1][invalid_writes] = 0

        # Linearize ground-plane indices (linear idx = y * W + x)
        linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
        linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
        linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()
        linear_locs_ss = linear_locs_ss[..., ~invalid_writes.reshape(-1)]
        tgt_img = img.reshape(1, f, -1)[..., ~invalid_writes.reshape(-1)]

        if self.scatter_mode == 'max':
            proj_feats, _ = torch_scatter.scatter_max(
                                tgt_img,
                                linear_locs_ss,
                                dim=2,
                                dim_size=outh*outw,
                            )
        elif self.scatter_mode == 'min':
            proj_feats, _ = torch_scatter.scatter_min(
                                tgt_img,
                                linear_locs_ss,
                                dim=2,
                                dim_size=outh*outw,
                            )
        elif self.scatter_mode == 'mean':
            proj_feats = torch_scatter.scatter_mean(
                                tgt_img,
                                linear_locs_ss,
                                dim=2,
                                dim_size=outh*outw,
                            )
        else:
            raise ValueError("Invalid scatter mode!")

        proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)

        return proj_feats


class RotateTensor:
    def __init__(self, device):
        self.device = device

    def forward(self, x_gp, heading):
        sin_t = torch.sin(heading.squeeze(1))
        cos_t = torch.cos(heading.squeeze(1))
        A = torch.zeros(x_gp.size(0), 2, 3).to(self.device)
        A[:, 0, 0] = cos_t
        A[:, 0, 1] = sin_t
        A[:, 1, 0] = -sin_t
        A[:, 1, 1] = cos_t

        grid = F.affine_grid(A, x_gp.size())
        rotated_x_gp = F.grid_sample(x_gp, grid)
        return rotated_x_gp


class Projection:
    def __init__(self, egocentric_map_size, global_map_size, device, coordinate_min, coordinate_max, height_min, height_max, scatter_mode):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.compute_spatial_locs = ComputeSpatialLocs(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max, height_min, height_max
        )
        self.project_to_ground_plane = ProjectToGroundPlane(egocentric_map_size, device, scatter_mode)
        self.rotate_tensor = RotateTensor(device)

    def forward(self, img, depth, heading):
        spatial_locs, valid_inputs = self.compute_spatial_locs.forward(depth)
        x_gp = self.project_to_ground_plane.forward(img, spatial_locs, valid_inputs)
        rotated_x_gp = self.rotate_tensor.forward(x_gp, heading)
        return rotated_x_gp


class Registration:
    def __init__(self, egocentric_map_size, global_map_size, global_map_depth, device, coordinate_min, coordinate_max, num_obs):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth
        self.device = device
        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.num_obs = num_obs

    def forward(self, observations, full_global_map, egocentric_map):
        """
        Register egocentric_map to full_global_map

        Args:
            observations: Dictionary containing habitat observations
            full_global_map: (self.num_obs, self.global_map_size, self.global_map_size, self.global_map_depth) torch.tensor containing global map
            egocentric_map: (self.num_obs, self.egocentric_map_size, self.egocentric_map_size, self.global_map_depth) torch.tensor containing egocentrc map

        Returns:
            registered_map: (self.num_obs, self.global_map_size, self.global_map_size, self.global_map_depth) torch.tensor containing registered map
        """
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                agent_view = torch.cuda.FloatTensor(self.num_obs, self.global_map_depth, self.global_map_size, self.global_map_size).fill_(0)
        else:
            agent_view = torch.FloatTensor(self.num_obs, self.global_map_depth, self.global_map_size, self.global_map_size).to(self.device).fill_(0)

        agent_view[:, :, 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
        ] = egocentric_map

        st_pose = torch.cat(
            [-(grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
            -(grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
            observations['compass']], 
            dim=1
        )
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
        rotated = F.grid_sample(agent_view, rot_mat)
        translated = F.grid_sample(rotated, trans_mat)
        
        registered_map = torch.max(full_global_map, translated.permute(0, 2, 3, 1))

        return registered_map


class GlobalMap2World:
    # TODO: Needs testing
    # Class for converting global map indices to world coordinates
    def __init__(self, orig_position, orig_rot, grid_mapper: to_grid):
        self.device = orig_position.device
        self.orig_position = orig_position  # (3, ) torch tensor containing agent start position
        self.orig_rot = orig_rot  # (1, ) torch.tensor containing agent start rotation
        self.rot_mat = torch.tensor([[math.cos(orig_rot), -math.sin(orig_rot)], [math.sin(orig_rot), math.cos(orig_rot)]], device=self.device)
        self.grid_mapper = grid_mapper  # Maps GPS coordinate to indices

    def convert(self, global_map_idx):
        # Global map to world position
        gps_x, gps_y = self.grid_mapper.get_gps_coords(global_map_idx)
        gps_tmp = self.rot_mat.T @ torch.tensor([[gps_x], [gps_y]])
        return torch.tensor([gps_tmp[0], self.orig_position[1], gps_tmp[-1]])

    def inv_convert(self, world_position):
        # World position to global map index
        import pdb; pdb.set_trace()
        gps_world_position = (world_position - self.orig_position)  # (2, ) torch tensor containing agent-centric position
        gps_world_position = torch.tensor([[gps_world_position[0, 0]], [gps_world_position[0, 2]]], device=self.device)
        gps_world_position = (self.rot_mat @ gps_world_position).T  # (1, 2)
        grid_x, grid_y = self.grid_mapper.get_grid_coords(gps_world_position)
        return grid_x.long().item(), grid_y.long().item()
