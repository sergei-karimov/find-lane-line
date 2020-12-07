# importing some useful packages
import matplotlib.image as mpimg
import numpy as np
from numpy import linalg
import cv2
import imageio
from moviepy.editor import VideoFileClip


class LaneLine(object):
    def extrapolite_lines(self, height, width, left_k, left_b, right_k, right_b):
        y1 = int(height * 0.99)
        y2 = int(height * 0.6)

        try:
            left_x1 = int((y1 - left_b) / left_k)
            left_x2 = int((y2 - left_b) / left_k)

            right_x1 = int((y1 - right_b) / right_k)
            right_x2 = int((y2 - right_b) / right_k)
        except:
            print('EXCEPTION: ', height, width, left_k, left_b, right_k, right_b)
            y1 = 0
            y2 = 0
            left_x1 = int(0)
            left_x2 = int(0)

            right_x1 = int(0)
            right_x2 = int(0)

        left_line = [(left_x1, y1), (left_x2, y2)]
        right_line = [(right_x1, y1), (right_x2, y2)]

        return left_line, right_line

    # to convert image to HLS color space
    def convert_hls(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # image is expected be in RGB color space
    def get_white_image(self, image):
        converted = self.convert_hls(image)

        # to find a white lane lines
        lower = np.uint8([0, 200, 0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(converted, lower, upper)

        # to find an yellow lane lines
        lower = np.uint8([10, 0, 100])
        upper = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(converted, lower, upper)

        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked

    # to convert image to grayscale color space
    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=13):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """

        left_b = []
        left_k = []
        right_b = []
        right_k = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # ignore a vertical line
                if y2 == y1:
                    continue  # ignore a horizontal line

                A = np.array([[x1, 1], [x2, 1]])
                c = np.array([y1, y2])

                # to find 'K' and 'b'
                k, b = linalg.solve(A, c)
                if np.abs(k) < 0.5:  # sets a some threshold
                    continue

                # length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if k < 0:
                    left_b.append(b)
                    left_k.append(k)
                elif k > 0:
                    right_b.append(b)
                    right_k.append(k)
        height, width, _ = img.shape
        left_line, right_line = self.extrapolite_lines(
            height,
            width,
            np.median(left_k),
            np.median(left_b),
            np.median(right_k),
            np.median(right_b)
        )

        cv2.line(img, left_line[0], left_line[1], color, thickness)
        cv2.line(img, right_line[0], right_line[1], color, thickness)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(
            img,
            rho,
            theta,
            threshold,
            np.array([]),
            minLineLength=min_line_len,
            maxLineGap=max_line_gap
        )

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img

    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def image_processing(self, image):
        img = np.zeros([100, 100, 3])
        if type(image) == str:
            img = mpimg.imread(image)
        else:
            img = np.copy(image)

        we_image = self.get_white_image(img)
        #    return we_image
        gs_image = self.grayscale(we_image)
        #    return gs_image
        gaussian_image = self.gaussian_blur(gs_image, 15)
        #    return gaussian_image
        canny_image = self.canny(gaussian_image, 50, 150)
        #    return canny_image
        height_image, width_image = canny_image.shape
        interested_zone = [np.array([[width_image * 0.01, height_image * 0.99],
                                     [width_image * 0.46, height_image * 0.6],
                                     [width_image * 0.57, height_image * 0.6],
                                     [width_image * 0.99, height_image * 0.99]], np.int32)]
        selected_image = self.region_of_interest(canny_image, interested_zone)
        #    return selected_image
        hough_image = self.hough_lines(
            selected_image,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            min_line_len=20,
            max_line_gap=300
        )

        #    return hough_image
        final_image = self.weighted_img(hough_image, img, 0.8, 1, 0.0)
        return final_image

    def video_processing(self, video_file_name, input_dir, output_dir):
        clip1 = VideoFileClip(input_dir + video_file_name)
        white_clip = clip1.fl_image(self.image_processing)  # NOTE: this function expects color images!!
        white_clip.write_videofile(output_dir + video_file_name, audio=False)
