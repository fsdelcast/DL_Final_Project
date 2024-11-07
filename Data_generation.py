import os

import mediapipe as mp # install mediapipe
import cv2 # install OpenCV library
import matplotlib.pyplot as plt
#from google.colab import drive

import csv
import copy
import argparse
import itertools


# Define the path where all the images are in 
DATA_DIR = '/content/drive/MyDrive/Deep_Learning/Final_Project/data'  # Replace 'data' with the actual folder name if different
# Define the path where you want the csv to be saved
csv_path = '/content/drive/MyDrive/Deep_Learning/Final_Project/landmarks.csv'

# import the hands mediapipe model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.35)


# Define the function to get the landmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


# Preprocessing of landmarks. Basically normalization, min-max
# My min max normalization
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    aux_x = []
    aux_y = []

    # Convert to relative coordinates
    for landmark_point in temp_landmark_list:
        current_x, current_y = landmark_point[0], landmark_point[1]
        aux_x.append(current_x)
        aux_y.append(current_y)


    min_x, min_y = min(aux_x), min(aux_y)
    max_x, max_y = max(aux_x), max(aux_y)

    def normalize_(n, min_value, max_value):
        return (n - min_value) / (max_value - min_value)

    x_normalized = list(map(lambda x: normalize_(x, min_x, max_x), aux_x))
    y_normalized = list(map(lambda x: normalize_(x, min_y, max_y), aux_y))

    #print(x_normalized)
    #print(y_normalized)
    final_list = [cor for pair in zip(x_normalized, y_normalized) for cor in pair]

    return final_list


# Write a csv
def logging_csv(label, landmark_list,csv_path):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ *landmark_list, label])
    return


# ---- this is important ----

# only run this once because this creates a new row with headers
header = [
    "1_X", "1_Y", "2_X", "2_Y", "3_X", "3_Y", "4_X", "4_Y", "5_X", "5_Y",
    "6_X", "6_Y", "7_X", "7_Y", "8_X", "8_Y", "9_X", "9_Y", "10_X", "10_Y",
    "11_X", "11_Y", "12_X", "12_Y", "13_X", "13_Y", "14_X", "14_Y", "15_X", "15_Y",
    "16_X", "16_Y", "17_X", "17_Y", "18_X", "18_Y", "19_X", "19_Y", "20_X", "20_Y",
    "21_X", "21_Y", "Label"
]

with open(csv_path, 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# ---- now with the good part ----

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): # open the data directory

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb) # get landmarks

        if result.multi_hand_landmarks:
          for hand_landmarks in result.multi_hand_landmarks:

            landmark_list = calc_landmark_list(img_rgb, hand_landmarks)
            #print(landmark_list)

            pre_processed_landmark = pre_process_landmark(landmark_list) # preprocess landmarks
            #print(pre_processed_landmark)

            logging_csv(dir_, pre_processed_landmark, csv_path) # write a csv
