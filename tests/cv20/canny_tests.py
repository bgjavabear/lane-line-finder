import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

test_image = mpimg.imread('resources/margo_test.jpg')
test_image_gray_scaled = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

test_image_blurred = cv2.GaussianBlur(test_image_gray_scaled, (5, 5), 1.4)

expected_image_with_blur = cv2.Canny(test_image_blurred, 20, 40)
expected_image_without_blur = cv2.Canny(test_image_gray_scaled, 20, 40)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.set_title('Initial')
ax1.imshow(test_image_gray_scaled, cmap='gray')
ax2.set_title('Canny without blur')
ax2.imshow(expected_image_without_blur, cmap='gray')
ax3.set_title('Canny with blur')
ax3.imshow(expected_image_with_blur, cmap='gray')

plt.show()
