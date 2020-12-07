import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from find_lane_line import LaneLine

OUTPUT_IMAGE_PATH = 'test_images_output/'
INPUT_IMAGE_PATH = "test_images/"

OUTPUT_VIDEO_PATH = 'test_videos_output/'
INPUT_VIDEO_PATH = "test_videos/"


def print_statistics(image):
    print('This image is:', type(image), 'with dimensions:', image.shape)


def plot_image(image, is_gray=False):
    # if you wanted to show a single color channel image called 'gray',
    # for example, call as plt.imshow(gray, cmap='gray')

    plt.figure(figsize=(18, 11))
    if is_gray:
        plt.imshow(image, aspect='auto', is_gray='gray')
    else:
        plt.imshow(image, aspect='auto')


def images_processing():
    img_list = os.listdir(INPUT_IMAGE_PATH)
    for image in img_list:
        processing = LaneLine()
        img = processing.image_processing(INPUT_IMAGE_PATH + image)

        plt.imshow(img, cmap='gray')

        file_name = OUTPUT_IMAGE_PATH + os.path.basename(image)
        mpimg.imsave(file_name, img)


def video_processing():
    processing = LaneLine()
    processing.video_processing("challenge.mp4", INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)


if __name__ == '__main__':
    # images_processing()
    video_processing()
