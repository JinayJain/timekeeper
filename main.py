from detection import detect_clock
from analysis import tell_time
import sys
import cv2

if len(sys.argv) != 2:
    print("Usage: python main.py <image_path>")
    exit(1)

# Change this to detect on another image
path = sys.argv[1]

img = cv2.imread(path)

bbox = detect_clock(img)

if bbox is None:
    print("No clock was found in this image")
    exit(0)

# print(f"Predicted bounding box: {bbox}")

img_boxed = img.copy()
cv2.rectangle(img_boxed, (bbox[0], bbox[1]),
              (bbox[2], bbox[3]), (255, 0, 0), 5)
img_boxed = cv2.putText(img_boxed, "clock", (
    bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, bottomLeftOrigin=False)

# cv2.imshow('image', img_boxed)

cv2.imshow('clock', img_boxed)
cv2.waitKey(0)

clock = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
clock = cv2.resize(clock, (400, 400))

cv2.imshow('clock', clock)
cv2.waitKey(0)

hours, minutes = tell_time(clock)

time_string = f"{hours}:{minutes:02}"
cv2.putText(clock, time_string, (220, 380),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 50, 50), 6)
cv2.rectangle(clock, (0, 0), (400, 400), (0, 0, 0), 3)

print(time_string)

img[:400, :400, :] = clock

cv2.imshow("clock", clock)
cv2.imwrite("preds.png", clock)

while 1:
    if cv2.waitKey(50) == ord('q'):
        break
cv2.destroyAllWindows()
