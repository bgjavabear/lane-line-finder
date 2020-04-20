import cv2
import cv20
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import utils

test_image = mpimg.imread('resources/solidWhiteCurve.jpg')
test_image_gray_scaled = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

expected_smoothed_image = cv2.GaussianBlur(test_image_gray_scaled, (5, 5), 1)
actual_smoothed_image = cv20.GaussianBlur(test_image_gray_scaled, 5, 1)

print('Before diff = ', utils.squared_diff(expected_smoothed_image, test_image_gray_scaled))
print('After diff = ', utils.squared_diff(expected_smoothed_image, actual_smoothed_image))

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.set_title('Expected')
ax1.imshow(expected_smoothed_image, cmap='gray')
ax2.set_title('Actual')
ax2.imshow(actual_smoothed_image, cmap='gray')
ax3.set_title('Before')
ax3.imshow(test_image_gray_scaled, cmap='gray')
plt.show()
