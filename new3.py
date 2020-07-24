import os
import time
import pprint
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from lib.core.evaluation import decode_preds
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from mtcnn.mtcnn import MTCNN
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
import cv2
import math
import numpy as np
import dlib
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from object_extractor import Extractor, FRONTALFACE_ALT2

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)
    parser.add_argument("-i", "--input", required=True)

    args = parser.parse_args()
    update_config(config, args)
    return args


args = parse_args()

logger, final_output_dir, tb_log_dir = \
utils.create_logger(config, args.cfg, 'test')

logger.info(pprint.pformat(args))
logger.info(pprint.pformat(config))

cudnn.benchmark = config.CUDNN.BENCHMARK
cudnn.determinstic = config.CUDNN.DETERMINISTIC
cudnn.enabled = config.CUDNN.ENABLED

config.defrost()
config.MODEL.INIT_WEIGHTS = False
config.freeze()
model = models.get_face_alignment_net(config)
print(model)


state_dict = torch.load(args.model_file,map_location=torch.device('cpu'))

if 'state_dict' in state_dict.keys():
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

dataset_type = get_dataset(config)

mean = np.asarray([0.4465, 0.4822, 0.4914])
std = np.asarray([0.1994, 0.1994, 0.2023])
out_size = 256

CURRENT_PATH = os.path.dirname(__file__)
extensions = ['jpeg', 'png', 'jpg']
out_src = './input/images/'
cap = cv2.VideoCapture(args.input)
index=1
frameRate = cap.get(5)

detector = MTCNN()

while(cap.isOpened()):
    frameId = cap.get(1)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        result = detector.detect_faces(frame)
        if result != []:
            for person in result:
                bounding_box = person['box']
                keypoints = person['keypoints']
    
                cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
                print(frame)
                crop=frame[bounding_box[1]: (bounding_box[1] + bounding_box[3]), bounding_box[0]: (bounding_box[0]+bounding_box[2])]
                print(crop)
                cv2.imwrite("face.jpg",crop)
                img = cv2.resize(crop,(256,256))
                raw_img = img
                img = img/255.0
                img = (img-mean)/std
                img = img.transpose((2, 0, 1))
                img = img.reshape((1,) + img.shape)
                input = torch.from_numpy(img).float()
                input= torch.autograd.Variable(input)
                out = model(input).cpu()
                out2=decode_preds(out, [(128,128)], [1.25], [64, 64])
                out2 = out2[0].numpy()

                for (x, y) in out2:
                    cv2.circle(raw_img, (x, y), 1, (0, 0, 255), -1)
                cv2.imwrite("./output/face_"+str(index)+".jpg", raw_img)
                index = index + 1
'''
        hog_face_detector = dlib.get_frontal_face_detector() 
        faces_hog = hog_face_detector(frame, 1)
        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        crop = frame[face.top():face.bottom(), face.left():face.right()]
        img = cv2.resize(crop,(256,256))
        raw_img = img
        img = img/255.0
        img = (img-mean)/std
        img = img.transpose((2, 0, 1))
        img = img.reshape((1,) + img.shape)
        input = torch.from_numpy(img).float()
        input= torch.autograd.Variable(input)
        out = model(input).cpu()
        out2=decode_preds(out, [(128,128)], [1.25], [64, 64])
        out2 = out2[0].numpy()

        for (x, y) in out2:
            cv2.circle(raw_img, (x, y), 1, (0, 0, 255), -1)
        cv2.imwrite("./output/face_"+str(index)+".jpg", raw_img)
        index = index + 1

'''
