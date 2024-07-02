import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Use GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply threshold to get a binary image
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours of each character
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from left to right, top to bottom
contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

# Loop through each contour and save corresponding character as separate image
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    char_image = image[y:y+h, x:x+w]
    cv2.imwrite(f'{i}.jpg', char_image)

print("Characters saved as individual images.")