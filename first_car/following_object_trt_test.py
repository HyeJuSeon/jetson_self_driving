gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

video_path = '../record/line.mp4'
import cv2
import numpy as np
from Adafruit_MotorHAT import Adafruit_MotorHAT

import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet, resnet18
import torch
from torch.nn import Linear
import torch.nn.functional as F
import PIL.Image

from torch2trt import TRTModule 


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

mean2 = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = transforms.Normalize(mean2, stdev)

def object_detection_preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

def is_blocked(frame):
    x = object_detection_preprocess(frame)
    y = alexnet(x)
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])
    if prob_blocked > 0.5:
        return True
    return False

def following_preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
	
angle = 0.0
angle_last = 0.0
#speed_gain_value = 1.0
#speed_gain_value = 0.18
speed_gain_value = 0.25
steering_value = 0.0001
#steering_gain_value = 1.0
#steering_dgain_value = 0.03
steering_gain_value = 0.03
steering_dgain_value = 0.0
steering_bias_value = 0.0
def execute_model(image):
    if is_blocked(image):
        all_stop()
        return
    global angle, angle_last

    xy = resnet(following_preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = 0.12

    print('resnet x %f, y %f' % (x, y))
   
    speed_value = speed_gain_value

    angle = np.arctan2(x, y)
    pid = angle * steering_gain_value + (angle - angle_last) * steering_dgain_value
    angle_last = angle

    steering_value = pid + steering_bias_value
    left_motor_value = max(min(speed_value + steering_value, 1.0), 0.0)
    right_motor_value = max(min(speed_value - steering_value, 1.0), 0.0)
    print('speed=', speed_value, ' angle=', angle, ' sterring=', steering_value)
   
    set_speed(motor_left_ID,   left_motor_value)
    set_speed(motor_right_ID,  right_motor_value)
    print('motor left=', left_motor_value, ' right=', right_motor_value)

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
		motor.run(Adafruit_MotorHAT.BACKWARD)
	else:
		motor.run(Adafruit_MotorHAT.FORWARD )

def all_stop():
	motor_left.setSpeed(0)
	motor_right.setSpeed(0)

	motor_left.run(Adafruit_MotorHAT.RELEASE)
	motor_right.run(Adafruit_MotorHAT.RELEASE)
	
def image_processing(output):
    execute_model(output)
    return output

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
    print(width)
    print(height)
    print('--' * 20)

    out = None
    # if savepath is not None:
    #     out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow('Input', cv2.WINDOW_GUI_EXPANDED)
    # cv2.namedWindow('Output', cv2.WINDOW_GUI_EXPANDED)
    linecolor1 = (0,240,240)
    linecolor2 = (230,0,0)

    try:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Our operations on the frame come here
                frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                output = image_processing(frame)		
                cv2.imshow('Input', frame)			
            else:
                break

            if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:  
        all_stop()
        cap.release()
        cv2.destroyAllWindows()
        return

    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()
    all_stop()
    cv2.destroyAllWindows()
    return
   
if __name__ == '__main__':
    resnet = resnet18(pretrained=False)
    resnet.fc = Linear(512, 2)
    resnet_trt = TRTModule()
    resnet_trt.load_state_dict(torch.load('best_resnet_trt.pth'))
    device = torch.device('cuda')
    resnet = resnet_trt.to(device)
    resnet = resnet_trt.eval()

    alexnet = alexnet(pretrained=False)
    alexnet.classifier[6] = Linear(alexnet.classifier[6].in_features, 2)
    alexnet.load_state_dict(torch.load('best_alexnet.pth'))
    print('best_alexnet.pth')
    alexnet = alexnet.to(device)

    motor_driver = Adafruit_MotorHAT(i2c_bus=1)

    motor_left_ID = 1
    motor_right_ID = 2

    motor_left = motor_driver.getMotor(motor_left_ID)
    motor_right = motor_driver.getMotor(motor_right_ID)
    # speed_value = speed_gain_value
    # angle = 0
    # pid = angle * steering_value + (angle - angle_last) * steering_dgain_value
    # angle_last = angle
    # steering_value = pid + steering_bias_value
    # print(steering_value)
    print('motor_driver ready')
    video(gst_str)