import matplotlib.image as mpimg
import image_processing as improc
import os
import cv2
from moviepy.editor import VideoFileClip

image_test_path = 'data/test_images/'
image_output_path = 'data/test_images_output/'
test_image_names = os.listdir(image_test_path)

for test_image_name in test_image_names:
    image_path = os.path.join(image_test_path, test_image_name)
    src = mpimg.imread(image_path)
    out = improc.finding_lane_lines(src)
    cv2.imwrite(os.path.join(image_output_path, test_image_name), out)

video_test_path = 'data/test_videos/'
video_output_path = 'data/test_videos_output/'
test_video_names = os.listdir(video_test_path)

for test_video_name in test_video_names:
    video_path = os.path.join(video_test_path, test_video_name)
    clip = VideoFileClip(video_path)
    out_clip = clip.fl_image(improc.finding_lane_lines)
    out_clip.write_videofile(os.path.join(video_output_path, test_video_name), audio=False)
