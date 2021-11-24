gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)224, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

import cv2
import numpy as np
import time
from Adafruit_MotorHAT import Adafruit_MotorHAT

import torch
from torch.nn import Linear
import torchvision
from torchvision.models import alexnet
import torch.nn.functional as F

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

def image_processing(frame):
    x = preprocess(frame)
    y = model(x)
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])
    if prob_blocked < 0.5:
        set_speed(motor_left_ID,   0.5)
        set_speed(motor_right_ID,  0.5)
    else :
        all_stop()
    return

def set_speed(motor_ID, value):
	max_pwm = 115.0
	speed = int(min(max(abs(value * max_pwm), 0), max_pwm))
	if motor_ID == 1:
		motor = motor_left
	elif motor_ID == 2:
		motor = motor_right
	else:
		return
	motor.setSpeed(speed)
	if value > 0:
		motor.run(Adafruit_MotorHAT.FORWARD)
	else:
		motor.run(Adafruit_MotorHAT.BACKWARD)

def all_stop():
	motor_left.setSpeed(0)
	motor_right.setSpeed(0)
	motor_left.run(Adafruit_MotorHAT.RELEASE)
	motor_right.run(Adafruit_MotorHAT.RELEASE)

def video(openpath, savepath = None):
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print('Video Opened')
    else:
        print('Video Not Opened')
        print('Program Abort')
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    #fourcc = cv2.VideoWriter_fourcc('m','p','4','v') with *.mp4 save
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image_processing(frame)
                cv2.imshow('Output', frame)
            else:
                break

            if cv2.waitKey(int(1000.0 / fps)) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:  
        all_stop()
        cap.release()
        cv2.destroyAllWindows()
        return
    # When everything done, release the capture
    all_stop()
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    model = alexnet(pretrained=False)
    model.classifier[6] = Linear(model.classifier[6].in_features, 2)
    print('alexnet')
    model.load_state_dict(torch.load('best_alexnet.pth'))
    print('best_alexnet.pth')
    device = torch.device('cuda')
    model = model.to(device)

    motor_driver = Adafruit_MotorHAT(i2c_bus=1)

    motor_left_ID = 1
    motor_right_ID = 2

    motor_left = motor_driver.getMotor(motor_left_ID)
    motor_right = motor_driver.getMotor(motor_right_ID)

    print('motor_driver ready')
    video(gst_str)
