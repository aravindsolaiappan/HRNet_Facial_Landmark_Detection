# HRNet_Facial_Landmark_Detection
#Prerequisites
download HR18-WFLW.pth pretrained model
#RUN
python Video_Landmark_detector.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml --model-file HR18-WFLW.pth -i input/videos/vid.avi
