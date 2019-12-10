import json
import os
# import cv2
import numpy as np

path = "/local/temporary/audi/camera/"
path_pic = "/local/temporary/audi/camera/camera/cam_front_center/"
def closest(lst, K):  
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

#with open('bus_signals.json', 'r') as f: #tady spise steering.json
with open(path +'steering.json', 'r') as f:
    data = json.load(f)

#data=sorted(data)
time_steering=list(data)

for i in range(len(time_steering)):
    time_steering[i] = int(time_steering[i])
#print(time_steering[1])
#1533906581461586
found_time=closest(time_steering,1533906581461587)
print(data[str(found_time)])
#for i in range(2):
    #print(data[i]['steering_angle_calculated'])

'''
new_data = {}
steering_data = {}
for i in data['steering_angle_calculated']['values']:
    steering_data[i[0]] = {}

for key, value in data.items():
    if key == 'steering_angle_calculated' or key == 'steering_angle_calculated_sign':
        for timestamp in value['values']:
            steering_data[timestamp[0]][key] = timestamp[1]


angle_timestamps = np.array([*steering_data])
'''
new_data = {}
for name in sorted(os.listdir(path_pic)):
    if name.endswith('.json'):

        with open(path_pic + name, 'r') as f:
            frame_json = json.load(f)
            #index = np.argmin(abs(frame_json['cam_tstamp'] - angle_timestamps))
            found_time=closest(time_steering, frame_json['cam_tstamp']) #finds closest timestamp in steerin.json to timestamp in image json
            new_data[frame_json['cam_tstamp']] = data[str(found_time)]  #priradi do dictionary new_data timestamp obrazku s daty jemu nejblizsimi
            # print(new_data)
            # print('\n')
            #print(found_time)
            #index = 

            #new_data[frame_json['cam_tstamp']] = steering_data[angle_timestamps[index]]

            #np.save('processed/' + frame_json['image_png'].split('_')[-1][:-4] + '.npy', frame)

# print(new_data)
#with open('steering.json', 'w') as g:
#    json.dump(new_data, g)

### save steering
# sequence 13 - highway with swapping lanes - 1. test

