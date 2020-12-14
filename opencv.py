import cv2
import numpy as np


def gray_to_rgb(gray):
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def rescale(img, factor):
    width = int(img.shape[0] * factor)
    height = int(img.shape[1] * factor)
    return cv2.resize(img, (width, height))

# TODO: Incorporate vertical stacking
def join_images(*images, horizontal=True):
    img_tuple = ()

    for img in images:
        if len(img.shape) == 2:
            img = gray_to_rgb(img)

        img_tuple = img_tuple + (img,)

    return np.hstack(img_tuple)


path = "./data/clocks/clock1.jpg"

# read and rescale the source image
img = cv2.imread(path)
img = rescale(img, 0.5)

# convert to grayscale and add blurring to reduce noise
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (9,9), 0)
gray_blur = cv2.medianBlur(gray, 7)

# preview canny detector used in circle detection
cannyThresh1, cannyThresh2 = 100, 200
canny = cv2.Canny(gray_blur, cannyThresh1, cannyThresh2)

# find best circle and mask grayscale image using it
circles = cv2.HoughCircles(
    gray_blur, cv2.HOUGH_GRADIENT, 1, 200, minRadius=100, maxRadius=300)
circles = np.uint16(np.around(circles))
best_circle = circles[0][0]
mask = np.zeros_like(gray_blur)
cv2.circle(mask, (best_circle[0], best_circle[1]), best_circle[2], 1, -1)
masked = gray * mask

thresh = cv2.adaptiveThreshold(
    masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.erode(thresh, kernel)
thresh = cv2.Canny(thresh, 100, 200)

cv2.imshow("image", join_images(img, gray, masked, canny, thresh))

while True:
    if cv2.waitKey(50) == ord('q'):
        break

cv2.destroyAllWindows()
