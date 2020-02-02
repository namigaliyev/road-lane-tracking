import cv2
import numpy as np
from collections import deque
from lane_detection import progress

height, width = 540, 960
test_video = 'data/videos/test.mp4'

cap = cv2.VideoCapture(test_video)

# create queue
frame_buffer = deque(maxlen=10)
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frame_buffer.append(frame)
        
        blend = progress(frames=frame_buffer)
        cv2.imshow('blend', cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))

        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()