import cv2
import os

# Horizontal flip
imageFolderPath = 'exercise_images'

inputImagePath = os.path.join(imageFolderPath, 'look_up_left.png')
outputImagePath = os.path.join(imageFolderPath, 'look_up_right.png')

originalImage = cv2.imread(inputImagePath)
flipImage = cv2.flip(originalImage, 1)
cv2.imshow('flip', flipImage)

# Press any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(outputImagePath, flipImage)