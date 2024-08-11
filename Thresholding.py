import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import cv2
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt


def Optimized_Thresholding(Input_Img, option):
    img_gray = cv2.cvtColor(Input_Img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img_gray, 5)  # applies median blur using medianBlur function to reduce noise.

    if option == 'Global':
        # The THRESH_BINARY flag indicates that pixels with intensity greater than the threshold will be set to 255 (white)
        ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

        # Plot the grayscale image
        plt.figure(figsize=(th1.shape[1] / 100, th1.shape[0] / 100))
        plt.imshow(th1, cmap='gray')  # Specify colormap as 'gray' for grayscale image
        plt.axis("off")

        # Save the figure
        plt.savefig('./images/optimal_out1.png', bbox_inches='tight', pad_inches=0)
        return th1

    if option == 'Local':
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Plot the grayscale image
        plt.figure(figsize=(th2.shape[1] / 100, th2.shape[0] / 100))
        plt.imshow(th2, cmap='gray')  # Specify colormap as 'gray' for grayscale image
        plt.axis("off")

        # Save the figure
        plt.savefig('./images/optimal_out2.png', bbox_inches='tight', pad_inches=0)

        return th2


def otsu_threshold(gray):
    """ The function calculates the optimal threshold value by iteratively
    evaluating different threshold values and selecting the one that maximizes the between-class variance,
    effectively separating the foreground and background regions of the image."""

    pixel_number = gray.shape[0] * gray.shape[1]  # number of pixels
    mean_weight = 1.0 / pixel_number  # sum of all weights
    his, bins = np.histogram(gray, np.arange(0, 257))  # calculating the histogram of pixel intensities

    final_thresh = -1  # defining the best threshold calculated
    final_variance = -1  # defining the highest between class variance
    intensity_arr = np.arange(256)  # creating array of all the possible pixel values (0-255)

    # Iterating through all the possible pixel values from the histogram as thresholds
    for t in bins[0:-1]:
        pcb = np.sum(his[:t])  # summing the frequency of the values before the threshold (background)
        pcf = np.sum(his[t:])  # summing the frequency of the values after the threshold (foreground)

        # if pcb == 0 or pcf == 0:  # Skip calculation if pcb or pcf is zero
        #     continue

        Wb = pcb * mean_weight  # calculating the weight of the background (divide the frequencies by the sum of all weights)
        Wf = pcf * mean_weight  # calculating the weight of the foreground

        # calculating the mean of the background (multiply the background
        # pixel value with its weight, then divide it with the sum of frequencies of the background)
        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)  # calculating the mean of the foreground

        variance = Wb * Wf * (mub - muf) ** 2  # calculate the between class variance

        if variance > final_variance:  # compare the variance in each step with the previous
            final_thresh = t
            final_variance = variance

    return final_thresh


def local_thresholding(image):
    # If the image is colored, change it to grayscale, otherwise take the image as it is
    if (image.ndim == 3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif (image.ndim == 2):
        gray = image

    height, width = gray.shape  # get the height and width of the image
    # In this case we will divide the image into a 2x2 grid image
    half_height = height // 2
    half_width = width // 2

    # Getting the four section of the 2x2 image, It calculates the coordinates to divide the image into four equal sections
    section_1 = gray[:half_height, :half_width]
    section_2 = gray[:half_height, half_width:]
    section_3 = gray[half_height:, :half_width]
    section_4 = gray[half_height:, half_width:]


    # Applying the threshold of each section on its corresponding section
    section_1[section_1 > otsu_threshold(section_1)] = 255
    section_1[section_1 < otsu_threshold(section_1)] = 0

    section_2[section_2 > otsu_threshold(section_2)] = 255
    section_2[section_2 < otsu_threshold(section_2)] = 0

    section_3[section_3 > otsu_threshold(section_3)] = 255
    section_3[section_3 < otsu_threshold(section_3)] = 0

    section_4[section_4 > otsu_threshold(section_4)] = 255
    section_4[section_4 < otsu_threshold(section_4)] = 0

    # Regroup the sections to form the final image
    top_section = np.concatenate((section_1, section_2), axis=1)
    bottom_section = np.concatenate((section_3, section_4), axis=1)
    final_img = np.concatenate((top_section, bottom_section), axis=0)

    plt.imshow(final_img, cmap='gray')
    path = "images/local.png"
    plt.axis("off")
    plt.savefig(path)
    return path


def global_thresholding(image):
    # If the image is colored, change it to grayscale, otherwise take the image as it is
    if (image.ndim == 3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif (image.ndim == 2):
        gray = image

    t = otsu_threshold(gray)

    # Applying the threshold on the image whether it is calculated
    final_img = gray.copy()
    final_img[gray > t] = 255
    final_img[gray < t] = 0

    plt.imshow(final_img, cmap='gray')
    path = "images/global.png"
    plt.axis("off")
    plt.savefig(path)
    return path



# these 2 functions is for task 1 
def local_thresholding_1(gray,slider_1,slider_2,slider_3,slider_4):
        """
        Apply thresholding to the image based on user-defined parameters.

        Args:
            image: Input image.
        """

        height, width = gray.shape  # get the height and width of the image
        # In this case we will divide the image into a 2x2 grid image
        half_height = height // 2
        half_width = width // 2

        # Getting the four section of the 2x2 image
        section_1 = gray[:half_height, :half_width]
        section_2 = gray[:half_height, half_width:]
        section_3 = gray[half_height:, :half_width]
        section_4 = gray[half_height:, half_width:]

        section_1[section_1 > slider_1] = 255
        section_1[section_1 < slider_1] = 0

        section_2[section_2 > slider_2] = 255
        section_2[section_2 < slider_2] = 0

        section_3[section_3 > slider_3] = 255
        section_3[section_3 < slider_3] = 0

        section_4[section_4 >slider_4] = 255
        section_4[section_4 < slider_4] = 0

        # Regroup the sections to form the final image
        top_section = np.concatenate((section_1, section_2), axis=1)
        bottom_section = np.concatenate((section_3, section_4), axis=1)
        final_img = np.concatenate((top_section, bottom_section), axis=0)

        return final_img


def global_thresholding_1(gray,slider_1):

        final_img = gray.copy()
        final_img[gray > slider_1] = 255
        final_img[gray < slider_1] = 0
        return final_img





def spectral_threshold(img):
    """
     Applies Thresholding To The Given Grayscale Image Using Spectral Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    src = np.copy(img)

    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # Get Image Dimensions
    YRange, XRange = src.shape
    # Get The Values of The Histogram Bins
    HistValues = plt.hist(src.ravel(), 256)[0]
    # Calculate The Probability Density Function
    PDF = HistValues / (YRange * XRange)
    # Calculate The Cumulative Density Function
    CDF = np.cumsum(PDF)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    Global = np.arange(0, 256)
    GMean = sum(Global * PDF) / CDF[-1]
    for LowT in range(1, 254):
        for HighT in range(LowT + 1, 255):
            try:
                # Background Intensities Array
                Back = np.arange(0, LowT)
                # Low Intensities Array
                Low = np.arange(LowT, HighT)
                # High Intensities Array
                High = np.arange(HighT, 256)
                # Get Low Intensities CDF
                CDFL = np.sum(PDF[LowT:HighT])
                # Get Low Intensities CDF
                CDFH = np.sum(PDF[HighT:256])
                # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
                BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
                LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
                HighMean = sum(High * PDF[HighT:256]) / CDFH
                # Calculate Cross-Class Variance
                Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                        CDFH * (HighMean - GMean) ** 2))
                # Filter Out Max Variance & It's Threshold
                if Variance > MaxVariance:
                    MaxVariance = Variance
                    OptimalLow = LowT
                    OptimalHigh = HighT
            except RuntimeWarning:
                pass
    binary = np.zeros(src.shape, dtype=np.uint8)
    binary[src < OptimalLow] = 0
    binary[(src >= OptimalLow) & (src < OptimalHigh)] = 128
    binary[src >= OptimalHigh] = 255
    return binary



def local_spectral_thresholding(image, t1, t2, t3, t4, mode):
    # If the image is colored, change it to grayscale, otherwise take the image as it is
    if (image.ndim == 3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif (image.ndim == 2):
        gray = image

    height, width = gray.shape # get the height and width of the image
    # In this case we will divide the image into a 2x2 grid image
    half_height = height//2 
    half_width = width//2

    # Getting the four section of the 2x2 image
    section_1 = gray[:half_height, :half_width]
    section_2 = gray[:half_height, half_width:]
    section_3 = gray[half_height:, :half_width]
    section_4 = gray[half_height:, half_width:]

    # Check if the threshold is calculated through Otsu's method or given by the user
    if (mode == 2): # calculating the threshold using Otsu's methond for each section
        t1 = spectral_threshold(section_1)
        t2 = spectral_threshold(section_2)
        t3 = spectral_threshold(section_3)
        t4 = spectral_threshold(section_4)

    # Applying the threshold of each section on its corresponding section
    section_1[section_1 > t1] = 255
    section_1[section_1 < t1] = 0

    section_2[section_2 > t2] = 255
    section_2[section_2 < t2] = 0

    section_3[section_3 > t3] = 255
    section_3[section_3 < t3] = 0

    section_4[section_4 > t4] = 255
    section_4[section_4 < t4] = 0

    # Regroup the sections to form the final image
    top_section = np.concatenate((section_1, section_2), axis = 1)
    bottom_section = np.concatenate((section_3, section_4), axis = 1)
    final_img = np.concatenate((top_section, bottom_section), axis=0)

        # final_img = gray.copy()
        # final_img[gray > t] = 255
        # final_img[gray < t] = 0

    print("hhhhhhhhhhhhhhhhhhhhhhhhh")
    
    return final_img
