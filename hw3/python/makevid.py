image_folder = '../result/vid2'
video_name = 'video2.avi'

import glob
import os

#
# check what files I don't have
dont = [i for i in range(511)]
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
for img in images:
    n = int(img.split('.')[0])
    dont.remove(n)
    # print(len(dont))
print(dont)

import cv2
import os

# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = [str(i) + '.jpg' for i in range(511) if i not in dont]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 20, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
