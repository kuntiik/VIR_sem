import json
import os
import cv2
import numpy as np

with open('bus_signals.json', 'r') as f:
    data = json.load(f)

new_data = {}
steering_data = {}
for i in data['steering_angle_calculated']['values']:
    steering_data[i[0]] = {}

for key, value in data.items():
    if key == 'steering_angle_calculated' or key == 'steering_angle_calculated_sign':
        for timestamp in value['values']:
            steering_data[timestamp[0]][key] = timestamp[1]


angle_timestamps = np.array([*steering_data])

new_data = {}
for name in sorted(os.listdir('cam_front_center')):
    if name.endswith('.json'):

        with open('cam_front_center/' + name, 'r') as f:
            frame_json = json.load(f)
            index = np.argmin(abs(frame_json['cam_tstamp'] - angle_timestamps))

            new_data[frame_json['cam_tstamp']] = steering_data[angle_timestamps[index]]

            #np.save('processed/' + frame_json['image_png'].split('_')[-1][:-4] + '.npy', frame)


with open('steering.json', 'w') as g:
    json.dump(new_data, g)

### save steering
# sequence 13 - highway with swapping lanes - 1. test

