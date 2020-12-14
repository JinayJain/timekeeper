import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def tell_time(img):
    processed = preprocess(img)
    canny = cv2.Canny(processed, 100, 200)

    lines = find_lines(canny)

    cx = img.shape[0] / 2
    cy = img.shape[1] / 2
    radius = 50

    hand_lines = find_hand_lines(lines, cx, cy, radius)

    clusters = cluster_lines(hand_lines)
    summary = summarize_clusters(clusters)

    if len(summary) == 1:  # the minute and hour hand might be clustered together if they overlap
        summary.append(summary[0])

    # print(f"Summary list {summary}")

    hours, minutes = time_from_angles(summary[1][1], summary[0][1])

    for line in lines:
        for x1, y1, x2, y2 in line:
            if line_near_center(x1, y1, x2, y2, cx, cy, radius):
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imshow("lines", img)
    # cv2.imshow("canny", canny)

    return hours, minutes


def preprocess(img):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)

    return img


def find_lines(canny):
    threshold = 50
    deg_resolution = np.pi / 180
    rad_resolution = 1
    min_length = 50
    max_line_gap = 10

    lines = cv2.HoughLinesP(canny, rad_resolution, deg_resolution,
                            threshold, minLineLength=min_length, maxLineGap=max_line_gap)

    return lines


def dist(x1, y1, x2, y2):
    '''Euclidean distance between two points'''
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def line_near_center(x1, y1, x2, y2, cx, cy, radius):
    '''Determines whether a line segment has a point within a certain distance from the center'''
    return dist(x1, y1, cx, cy) <= radius or dist(x2, y2, cx, cy) <= radius


def find_hand_lines(lines, cx, cy, radius):
    """Finds the lines that are close to the center of the image and returns their angles,lengths"""
    hand_lines = []  # list of tuples (angle, length)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if line_near_center(x1, y1, x2, y2, cx, cy, radius):
                # ensure the first point is closer to the center
                if dist(x1, y1, cx, cy) > dist(x2, y2, cx, cy):
                    x1, y1, x2, y2 = x2, y2, x1, y1

                # y1 is before y2 since OpenCV coordinates reverse the direction of y
                angle = math.degrees(math.atan2(x2 - x1, y1 - y2))
                if angle < 0:
                    angle += 360
                length = dist(x1, y1, x2, y2)

                hand_lines.append((angle, length))

    return hand_lines


def plot_line_data(lines):
    x_list, y_list = [], []
    for x, y in lines:
        x_list.append(x)
        y_list.append(y)

    plt.xlabel("angle")
    plt.ylabel("length")
    plt.scatter(x_list, y_list)
    plt.show()


def cluster_lines(lines):
    """Clusters lines by their angle similarity determined by max_angle_gap"""

    lines.sort()  # sorts by angle

    max_angle_gap = 5
    clusters = [[lines[0]]]

    for i in range(1, len(lines)):
        if abs(lines[i][0] - lines[i-1][0]) <= max_angle_gap:
            clusters[len(clusters) - 1].append(lines[i])
        else:
            clusters.append([lines[i]])

    return clusters


def summarize_clusters(clusters):
    """Summarizes the clusters from cluster_lines into a sorted array of tuples, sorted by decreasing length

        [(len1, angle1), (len2, angle2), ...]"""

    summary = []

    for cluster in clusters:
        angles = np.array([angle for angle, length in cluster])
        lengths = np.array([length for angle, length in cluster])

        avg_angle = np.mean(angles)
        max_len = np.max(lengths)

        summary.append((max_len, avg_angle))

    summary.sort(reverse=True)

    return summary


def time_from_angles(hour_angle, minute_angle):
    # you could do some intelligent estimation based on the progression of the hour hand through the hour vs the minute hand
    # if the length differences are miniscule
    hour_ratio = hour_angle / 360.
    minute_ratio = minute_angle / 360.

    hours = hour_ratio * 12
    minutes = int(round(minute_ratio * 60)) % 60

    margin = 5
    # an attempt to reduce small angle errors in hours estimate
    if abs(minutes - 60) < 5 or minutes < 5:
        hours = int(round(hours))
    else:
        hours = math.floor(hours)

    if hours == 0:
        hours = 12


    return hours, minutes
