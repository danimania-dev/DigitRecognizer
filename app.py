import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from model import Model, trans_eval

import cv2
import numpy as np
import os

model = Model()
model.load_state_dict(torch.load("trained.pth"))
model.eval()

# Testing

model.test_model('test_data')
model.test_model('test_data_bogdan')

# Demo

SCREEN_SIZE = 280
FINAL_SIZE = 28
THICKNESS = 12

screen = np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype = np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing, screen 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(screen, (x, y), THICKNESS, 255, -1)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(screen, (x, y), THICKNESS, 255, -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Predictor')
cv2.setMouseCallback('Predictor', draw)

while True:
    cv2.imshow('Predictor', screen)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        sm_image = cv2.resize(screen, (FINAL_SIZE, FINAL_SIZE), interpolation = cv2.INTER_AREA)
        
        file_name = "last.png"
            
        cv2.imwrite(file_name, sm_image)
        
        screen = np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype = np.uint8)

        print(model.predict("last.png"))

        os.remove("last.png")
        
    elif key == ord('x') or key == ord('X'):
        screen = np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype = np.uint8)
        
    elif key == 27: 
        break

cv2.destroyAllWindows()
