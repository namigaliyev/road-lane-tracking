import matplotlib.pyplot as plt
import cv2
import numpy as np
from os.path import join, basename
from collections import deque

# grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# canny
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# gaussian blur
def gaussian_blur(img, kernel_size):
  return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# detect hough line
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines

# calculate slope
def get_slope(x1,y1,x2,y2):
    return (y2 - y1) / (x2 - x1 + np.finfo(float).eps)

# calculate bias
def get_bias(slope, x1, y1):
        return y1 - (slope * x1)

def draw_lines(line_img, line, color=[255, 0, 0], thickness=12):
    x1, y1, x2, y2 = line
    cv2.line(line_img, ((int(x1)), (int(y1))), ((int(x2)), (int(y2))), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)