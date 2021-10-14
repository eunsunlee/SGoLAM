import numpy as np
import cv2

def compute_hsv_metric_from_rgb(img, color):
    '''
    Args:
        img: numpy array of h x w x 3
            normalized rgb image [0, 1]
        color: numpy array of 3
            normalized rgb values from [0, 1]
    Returns:
        dist: numpy array of h x w
    '''
    hsv_img = get_norm_hsv_img(255 * img.astype(np.uint8))
    hsv_color = get_norm_hsv_img(255 * color.reshape(1, 1, 3).astype(np.uint8))
    hsv_dist = hsv2hsv_cone(hsv_img) - hsv2hsv_cone(hsv_color)
    hsv_dist = np.sqrt(np.sum(hsv_dist ** 2, axis=2))
    return hsv_dist

def hsv2hsv_cone(x):
    return np.stack([
        np.sin(x[:, :, 0]) * x[:, :, 1] * x[:, :, 2],
        np.cos(x[:, :, 0]) * x[:, :, 1] * x[:, :, 2],
        x[:, :, 2]
    ], axis=2)

def get_norm_hsv_img(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return np.stack([
        (hsv_img[:, :, 0] / 179.) * 2 * np.pi,
        hsv_img[:, :, 1] / 255.,
        hsv_img[:, :, 2] / 255.
    ], axis=2)
