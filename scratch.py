import cv2
import time

from tracker import EuclideanDistTracker

tracker = EuclideanDistTracker()
obj_det = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

WebcamIsUsing = False
if WebcamIsUsing:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(r"C:\Users\sd674\Desktop\Vehicle-Detection-And-Counting-using-OpenCV-main\highway.mp4")

# Define a variable to keep track of the previous frame's timestamp
prev_frame_time = 0

# Assuming the pixels to meters conversion factor is known (e.g., 0.1 pixels/meter)
pixels_to_meters = 0.1

while True:
    _, img = cap.read()

    h, w, _, = img.shape
    roi = img[340:720, 500:800]
    mask = obj_det.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    cont, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    det = []
    for cnt in cont:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            det.append([x, y, w, h])

    boxes_ids = tracker.update(det)

    # Calculate current frame's timestamp
    current_frame_time = time.time()

    # Calculate elapsed time since previous frame
    elapsed_time = current_frame_time - prev_frame_time

    # Update previous frame's timestamp
    prev_frame_time = current_frame_time

    # Calculate the speed for each detected object
    for box in boxes_ids:
        x, y, w, h, id = box

        # Calculate distance traveled (assuming constant speed along x-axis)
        distance = w * pixels_to_meters

        # Calculate speed (meters per second)
        speed = distance / elapsed_time

        # Display speed on the frame
        cv2.putText(roi, f" {speed:.2f} m/s", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("mask", mask)
    cv2.imshow("roi", roi)
    cv2.imshow("img", img)

    key = cv2.waitKey(30)
    if key == 113:  # 113=Q
        break

cap.release()
cv2.destroyAllWindows()
