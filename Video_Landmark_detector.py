import os
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
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
import cv2
import math
import numpy as np
import dlib
import matplotlib.pyplot as plt
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
model = nn.DataParallel(model)

state_dict = torch.load(args.model_file,map_location=torch.device('cpu'))
if 'state_dict' in state_dict.keys():
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
else:
    model.module.load_state_dict(state_dict)

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
while(cap.isOpened()):
    frameId = cap.get(1)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = "./input/face_"+str(index)+ ".jpg"
        cv2.imwrite(filename, frame)
        Extractor.extract(os.path.join(CURRENT_PATH, filename), cascade_file=FRONTALFACE_ALT2,
                          output_directory=os.path.join(CURRENT_PATH, out_src), output_prefix="face_" + str(index),
                          start_count=1)
        index=index+1
cap.release()


#Perfroming the image preprocessing, marking the facial landmarks and storing it in output
directory = './input/images'
out_src = './output'

index = 1

for dir_path, _, file_names in os.walk(directory):
    for filename in file_names:
        path = './' + os.path.relpath(os.path.join(dir_path, filename))
        if not filename.split('.')[-1] in extensions: continue
        print(path)
        img = cv2.imread(path)
        img = cv2.resize(img,(256,256))
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

