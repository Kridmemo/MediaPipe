#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:51:50 2023

@author: kridsumangsri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:21:54 2023

@author: kridsumangsri
"""

!pip install mediapipe opencv-python

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def check_pronator_drift(landmarks):
    # Extract the y-coordinates of the landmarks
    left_landmarks_y = [landmarks[i].y for i in [13, 15, 17, 19, 21]]
    right_landmarks_y = [landmarks[i].y for i in [14, 16, 18, 20, 22]]

    # Calculate the average y-coordinates for left and right landmarks
    left_avg_y = sum(left_landmarks_y) / len(left_landmarks_y)
    right_avg_y = sum(right_landmarks_y) / len(right_landmarks_y)

    # Set a threshold for the difference between left and right landmarks
    threshold = 0.1

    if abs(left_avg_y - right_avg_y) < threshold:
        return "Pronator Drift Negative"
    elif left_avg_y < right_avg_y:
        return "Right Pronator Drift Positive"
    else:
        return "Left Pronator Drift Positive"
 
# Initialize the webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Draw the pose landmarks on the frame
        frame.flags.writeable = True
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check for pronator drift and display the result on the frame
        if results.pose_landmarks:
            drift_result = check_pronator_drift(results.pose_landmarks.landmark)
            cv2.putText(frame, drift_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('MediaPipe Pose', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

