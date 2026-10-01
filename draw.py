import cv2
import numpy as np
import os

SCREEN_SIZE = 280
FINAL_SIZE = 28
DRAW_THICKNESS = 12

screen = np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype = np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing, screen 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(screen, (x, y), DRAW_THICKNESS, 255, -1)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(screen, (x, y), DRAW_THICKNESS, 255, -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Dibujar')
cv2.setMouseCallback('Dibujar', draw)

pic_count = 0

while True:
    cv2.imshow('Dibujar', screen)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        sm_img = cv2.resize(screen, (FINAL_SIZE, FINAL_SIZE), interpolation = cv2.INTER_AREA)
        
        file_image = f"{pic_count}.png"
        while os.path.exists(file_image):
            pic_count += 1
            file_image = f"{pic_count}.png"
            
        cv2.imwrite(file_image, sm_img)
        print("Saved as " + str(file_image))
        pic_count += 1
        
        screen = np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype = np.uint8)
        
    elif key == ord('x') or key == ord('X'):
        screen = np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype = np.uint8)
        
    elif key == 27: 
        break

cv2.destroyAllWindows()
