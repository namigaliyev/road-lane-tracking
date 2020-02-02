import matplotlib.pyplot as plt
import cv2
import numpy as np
from os.path import join, basename
from collections import deque
from utils import grayscale, canny, gaussian_blur, hough_lines, get_slope, get_bias, draw_lines, weighted_img

# region of interest
def region_of_interest(image):

    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array( [[
                [3*width/4, 3*height/5],
                [width/4, 3*height/5],
                [40, height],
                [width - 40, height]
            ]], dtype=np.int32 )

    #defining a blank mask to start with
    mask = np.zeros_like(image)

    # #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count # (255, 255, 255)
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# compute candidate lines
# and return left and right line coordinates
def compute_candidates(canditates_lines, img_shape):
    global left_lane, right_line
    # separate candidate lines according to their slope
    pos_lines = [l for l in canditates_lines if l["slope"] > 0]
    neg_lines = [l for l in canditates_lines if l["slope"] < 0]

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([l["bias"] for l in neg_lines]).astype(int)
    neg_slope = np.median([l["slope"] for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = np.array([x1, y1, x2, y2])

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    pos_bias = np.median([l["bias"] for l in pos_lines]).astype(int)
    pos_slope = np.median([l["slope"] for l in pos_lines])
    x1, y1 = 0, pos_bias
    x2, y2 = np.int32(np.round((img_shape[0] - pos_bias) / pos_slope)), img_shape[0]
    right_lane = np.array([x1, y1, x2, y2])

    return [left_lane, right_lane]

def get_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform gaussian blur
    blur = gaussian_blur(img_gray, kernel_size=17)
    
    # perform edge detection
    canny_edges = canny(blur, low_threshold=50, high_threshold=70)
    
    detected_lines = hough_lines(img=canny_edges,
                                 rho=rho,
                                 theta=theta,
                                 threshold=threshold,
                                 min_line_len=min_line_len,
                                 max_line_gap=max_line_gap)
    
    
    candidate_lines = []
    for line in detected_lines:
        for x1, y1, x2, y2 in line:
            slope = get_slope(x1, y1, x2, y2)
            if 0.5 <= np.abs(slope) <= 2:
                candidate_lines.append({"slope": slope, 
                                        "bias": get_bias(slope, x1, y1)})

    lane_lines = compute_candidates(candidate_lines, img_gray.shape)

    return lane_lines

def last_lines_averages(lane_lines):
    avg_lt = np.zeros((len(lane_lines), 4))
    avg_lr = np.zeros((len(lane_lines), 4))

    for i in range(0, len(lane_lines)):
        # left line
        x1, y1, x2, y2 = lane_lines[i][0]
        avg_lt[i] += np.array([x1, y1, x2, y2])
        # right line
        x1, y1, x2, y2 = lane_lines[i][1]
        avg_lr[i] += np.array([x1, y1, x2, y2])

    return [np.mean(avg_lt, axis=0), np.mean(avg_lr, axis=0)]

def progress(frames):
    detected_lines = []

    # last 10 frames
    for i in range(0, len(frames)):
        # detect lines
        left_right_lines = get_lines(img=frames[i], 
                                    rho=2, 
                                    theta=np.pi/180, 
                                    threshold=1, 
                                    min_line_len=15, 
                                    max_line_gap=5)

        detected_lines.append(left_right_lines)

    # prepare empty mask on which lines are drawn
    line_img = np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)

    # last 10 frames line averages
    detected_lines = last_lines_averages(detected_lines)

    # draw lines
    for lane in detected_lines:
        draw_lines(line_img, lane)

    # region of interest
    masked_img = region_of_interest(line_img)

    # img_color = frames[-1] if is_videoclip else frames[0]
    result = weighted_img(masked_img, frames[-1])
    return result