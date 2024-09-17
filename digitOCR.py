import numpy as np
import pygame
import sys
from pygame.locals import *
from keras.models import load_model
import cv2

# Initialize constants
sizex = 640
sizey = 480
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
boundry = 5
img_count = 1
image_save = False
predict = True

# Load the model
model = load_model("C:/Users/Dareen/OneDrive/Desktop/handwritten/digits.h5")
labels = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
          5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

# Initialize Pygame
pygame.init()
displaysurface = pygame.display.set_mode((sizex, sizey))
pygame.display.set_caption('Digits Board')
font = pygame.font.Font(None, 36)

# Variables for drawing
iswriting = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(displaysurface, white, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
            number_xcord = []
            number_ycord = []

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                rect_min_x = max(min(number_xcord) - boundry, 0)
                rect_max_x = min(max(number_xcord) + boundry, sizex)
                rect_min_y = max(min(number_ycord) - boundry, 0)
                rect_max_y = min(max(number_ycord) + boundry, sizey)

                # Capture the drawn area
                img_arr = np.array(pygame.surfarray.pixels3d(displaysurface))
                ing_arr = img_arr[rect_min_x:rect_max_x, rect_min_y:rect_max_y].mean(axis=2)

                # Process the image
                image = cv2.resize(ing_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255.0

                # Make prediction
                label = str(labels[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])
                text = font.render(label, True, red, white)
                displaysurface.blit(text, (10, 10))
                
                # Clear the screen after a short delay
                pygame.display.update()
                pygame.time.delay(2000)  # Delay for 2 seconds to view the prediction
                displaysurface.fill(black)  # Clear the screen

        # Save image if needed
        if image_save:
            cv2.imwrite(f'img_{img_count}.png', ing_arr)
            img_count += 1

    # Update the display
    pygame.display.update()