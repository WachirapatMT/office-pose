import cv2

# Horizontal flip
imageFolderPath = 'exercise_images'

inputImagePath = f'{imageFolderPath}/neck_bend_r.png'
outputImagePath = f'{imageFolderPath}/neck_bend_l.png'

originalImage = cv2.imread(inputImagePath)
flipImage = cv2.flip(originalImage, 1)
cv2.imshow('flip', flipImage)

# Press any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(outputImagePath, flipImage)