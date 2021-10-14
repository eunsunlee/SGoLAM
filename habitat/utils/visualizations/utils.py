#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm

from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps

cv2 = try_cv2_import()


def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view

def draw_subsuccess(view: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    r"""Draw translucent blue strips on the border of input view to indicate
    a subsuccess event has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of blue collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([0, 0, 255]) + (1.0 - alpha) * view)[mask]
    return view


def draw_found(view: np.ndarray, alpha: float = 1) -> np.ndarray:
    r"""Draw translucent blue strips on the border of input view to indicate
    that a found action has been called.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of blue collision strip. 1 is completely non-transparent.
    Returns:
        A view with found action effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([0, 0, 255]) + (1.0 - alpha) * view)[mask]
    return view

def observations_to_image(observation: Dict, info: Dict, action: np.ndarray) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        action: action returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        observation_size = observation["depth"].shape[0]
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view.append(depth_map)
    
    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    if action[0] == 0:
        egocentric_view = draw_found(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = info["top_down_map"]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info["top_down_map"]["fog_of_war_mask"]
        )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 16,
        )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame


def observations_to_image_challenge(observation: Dict, info: Dict, action: np.ndarray) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().
    Modified version for challenge.

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        action: action returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        observation_size = observation["depth"].shape[0]
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        if len(depth_map.shape) == 3:
            depth_map = np.expand_dims(depth_map, axis=0)
        egocentric_view.append(depth_map)
    
    # draw feature map if info contains feature map information
    if info['feature_map'] is not None:
        assert len(info['feature_map'].shape) == 4
        assert info['agent_pos'] is not None
        # Visualize only the first three channels
        feature_map = cv2.resize(info['feature_map'].cpu().squeeze(0).numpy()[..., :3], dsize=(rgb.shape[2], rgb.shape[1]), 
        interpolation=cv2.INTER_NEAREST)
        # Modify agent position with respect to resized map
        mod_x = int(info['agent_pos'][0] * rgb.shape[2] / info['feature_map'].shape[2])
        mod_y = int(info['agent_pos'][1] * rgb.shape[1] / info['feature_map'].shape[1])
        # Mark agent
        feature_map = (np.expand_dims(feature_map, 0) * 255.0).astype(np.uint8)
        feature_map = maps.draw_agent(
            image=feature_map.squeeze(0),
            agent_center_coord=(mod_x, mod_y),
            agent_rotation=observation['compass'].item() + np.pi,
            agent_radius_px=5,
        )        
        egocentric_view.append(np.expand_dims(feature_map, 0))

    # draw occupancy map if info contains occupancy map information
    if info['occupancy_map'] is not None:
        assert len(info['occupancy_map'].shape) == 4
        assert info['agent_pos'] is not None
        occupancy_map = cv2.resize(info['occupancy_map'].cpu().squeeze(0).numpy(), dsize=(rgb.shape[2], rgb.shape[1]),
        interpolation=cv2.INTER_NEAREST)
        # Modify agent position with respect to resized map
        mod_x = int(info['agent_pos'][0] * rgb.shape[2] / info['feature_map'].shape[2])
        mod_y = int(info['agent_pos'][1] * rgb.shape[1] / info['feature_map'].shape[1])
        # Mark agent
        occupancy_map = occupancy_map.reshape(1, rgb.shape[1], rgb.shape[2], 1)
        occupancy_map = np.concatenate([occupancy_map] * 3, axis=-1)
        occupancy_map = (occupancy_map * 255.0).astype(np.uint8)
        occupancy_map = maps.draw_agent(
            image=occupancy_map.squeeze(0),
            agent_center_coord=(mod_x, mod_y),
            agent_rotation=observation['compass'].item() + np.pi,
            agent_radius_px=5,
        )
        egocentric_view.append(np.expand_dims(occupancy_map, 0))

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=2)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    if action[0] == 0:
        egocentric_view = draw_found(egocentric_view)

    frame = egocentric_view

    return frame


def observations_to_image_custum(observation: Dict, walker, action: np.ndarray) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().
    Modified version for agent customized visualization.

    Args:
        observation: observation returned from an environment step().
        walker: Habitat agent containing maps and other information.
        action: action returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    # Get spatial dimensions
    if "rgb" in observation:
        H = max(observation["rgb"].shape)
        W = max(observation["rgb"].shape)
    elif "depth" in observation:
        H = max(observation["depth"].shape)
        W = max(observation["depth"].shape)
    else:
        H, W = 275, 275  # Arbitrarily chosen for 'blind' agents

    # Visualize observations
    if "rgb" in observation and "rgb" in walker.visualize_list:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()
        if len(rgb.shape) == 3:
            rgb = np.expand_dims(rgb, axis=0)
        egocentric_view.append(rgb)

    if "depth" in observation and "depth" in walker.visualize_list:
        observation_size = observation["depth"].shape[0]
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        if len(depth_map.shape) == 3:
            depth_map = np.expand_dims(depth_map, axis=0)
        egocentric_view.append(depth_map) 

    map_keys = list(filter(lambda x: 'map' in x, walker.visualize_list))

    # Visualize maps
    for map_name in map_keys:
        vis_map = getattr(walker, map_name)  # Map attribute to be visualized
        assert len(vis_map.shape) == 4
        if vis_map.shape[-1] >= 3:
            # Visualize only the first three channels
            color_map = cv2.resize(vis_map.cpu().squeeze(0).numpy()[..., :3], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            # Modify agent position with respect to resized map
            mod_x = int(walker.agent_pos[0] * W / vis_map.shape[2])
            mod_y = int(walker.agent_pos[1] * H / vis_map.shape[1])
            # Mark agent
            color_map = (np.expand_dims(color_map, 0) * 255.0).astype(np.uint8)
            color_map = maps.draw_agent(
                image=color_map.squeeze(0),
                agent_center_coord=(mod_x, mod_y),
                agent_rotation=observation['compass'].item() + np.pi,
                agent_radius_px=5,
            )        
            egocentric_view.append(np.expand_dims(color_map, 0))

        elif vis_map.shape[-1] == 2:
            # Visualize only the first channel
            gray_map = cv2.resize(vis_map.cpu().squeeze(0).numpy()[..., 0], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            # Modify agent position with respect to resized map
            mod_x = int(walker.agent_pos[0] * W / vis_map.shape[2])
            mod_y = int(walker.agent_pos[1] * H / vis_map.shape[1])
            # Mark agent
            gray_map = gray_map.reshape(1, H, W, 1)
            gray_map = np.concatenate([gray_map] * 3, axis=-1)
            gray_map = (gray_map * 255.0).astype(np.uint8)
            gray_map = maps.draw_agent(
                image=gray_map.squeeze(0),
                agent_center_coord=(mod_x, mod_y),
                agent_rotation=observation['compass'].item() + np.pi,
                agent_radius_px=5,
            )
            egocentric_view.append(np.expand_dims(gray_map, 0))

        elif vis_map.shape[-1] == 1:
            gray_map = cv2.resize(vis_map.cpu().squeeze(0).numpy(), dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            # Modify agent position with respect to resized map
            mod_x = int(walker.agent_pos[0] * W / vis_map.shape[2])
            mod_y = int(walker.agent_pos[1] * H / vis_map.shape[1])
            # Mark agent
            gray_map = gray_map.reshape(1, H, W, 1)
            gray_map = np.concatenate([gray_map] * 3, axis=-1)
            gray_map = (gray_map * 255.0).astype(np.uint8)
            gray_map = maps.draw_agent(
                image=gray_map.squeeze(0),
                agent_center_coord=(mod_x, mod_y),
                agent_rotation=observation['compass'].item() + np.pi,
                agent_radius_px=5,
            )
            egocentric_view.append(np.expand_dims(gray_map, 0))
    
    # Visualize walker-specified images
    img_keys = list(filter(lambda x: 'img' in x, walker.visualize_list))
    for img_name in img_keys:
        vis_img = getattr(walker, img_name)  # Last dimension should be channel
        assert (vis_img.shape[-1] == 3 or vis_img.shape[-1] == 1) and len(vis_img.shape) == 3 # Only receives H * W * 3, H * W * 1 tensors
        if "torch" in str(type(vis_img)):
            vis_img = vis_img.cpu().float().numpy()
        elif "numpy" in str(type(vis_img)):
            vis_img = vis_img.astype(np.float)
        else:
            raise ValueError("Invalid input!")

        vis_img = cv2.resize(vis_img, dsize=(W, H))
        if vis_img.shape[-1] == 3:
            vis_img = (np.expand_dims(vis_img, 0) * 255.0).astype(np.uint8)  # Assume vis_img is in range [0, 1]
        else:
            vis_img = np.stack([vis_img] * 3, -1)
            vis_img = (np.expand_dims(vis_img, 0) * 255.0).astype(np.uint8)  # Assume vis_img is in range [0, 1]
        egocentric_view.append(vis_img)
    
    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=2)

    if action[0] == 0:
        egocentric_view = draw_found(egocentric_view)

    frame = egocentric_view

    return frame


def append_text_to_image(image: np.ndarray, text: str):
    r""" Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final
