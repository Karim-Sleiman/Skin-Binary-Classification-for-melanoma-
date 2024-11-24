import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from PIL import Image
import pywt

from monai.data import pad_list_data_collate
from skimage.filters import gabor, gabor_kernel
from skimage.util import img_as_ubyte
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import data, filters, color
from skimage.transform import radon, rescale
from scipy.stats import kurtosis, skew
from skimage.measure import regionprops, label
from scipy.spatial.distance import directed_hausdorff

def color_features(image):
    # BGR mean
    blue_mean = np.mean(image[:, :, 0])
    green_mean = np.mean(image[:, :, 1])
    red_mean = np.mean(image[:, :, 2])
    
    # BGR std
    blue_std = np.std(image[:, :, 0])
    green_std = np.std(image[:, :, 1])
    red_std = np.std(image[:, :, 2])
    
    # BGR skewness
    blue_skew = skew(image[:, :, 0].ravel())
    green_skew = skew(image[:, :, 1].ravel())
    red_skew = skew(image[:, :, 2].ravel())
    
    # LAB mean
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])
    
    # LAB std
    l_std = np.std(lab_image[:, :, 0])
    a_std = np.std(lab_image[:, :, 1])
    b_std = np.std(lab_image[:, :, 2])
    
    # LAB skewness
    l_skew = skew(lab_image[:, :, 0].ravel())
    a_skew = skew(lab_image[:, :, 1].ravel())
    b_skew = skew(lab_image[:, :, 2].ravel())
    
    # HSV mean
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv_image[:, :, 0])
    s_mean = np.mean(hsv_image[:, :, 1])
    v_mean = np.mean(hsv_image[:, :, 2])
    
    # HSV std
    h_std = np.std(hsv_image[:, :, 0])
    s_std = np.std(hsv_image[:, :, 1])
    v_std = np.std(hsv_image[:, :, 2])
    
    # HSV skewness
    h_skew = skew(hsv_image[:, :, 0].ravel())
    s_skew = skew(hsv_image[:, :, 1].ravel())
    v_skew = skew(hsv_image[:, :, 2].ravel())
    
    # Chromacity
    chromacity = s_mean / (v_mean + 1e-6)
    
    return (blue_mean, green_mean, red_mean, blue_std, green_std, red_std, blue_skew, green_skew, red_skew, 
            l_mean, a_mean, b_mean, l_std, a_std, b_std, l_skew, a_skew, b_skew, 
            h_mean, s_mean, v_mean, h_std, s_std, v_std, h_skew, s_skew, v_skew, 
            chromacity)

def Entropy(img):
    E = []
    for i in range(3):
        channel = img[:,:,i]
        n, bins = np.histogram(channel.ravel(), bins=256, range=[0, 256])
        e = 0
        total = channel.size
        for count in n:
            if count> 0:
                e += (count/total)*np.log2(count/total)
        E.append(-e)
    E = np.array(E).flatten()
    return E

def StdBGR(image):
    u, stddev = cv2.meanStdDev(image)
    stddev = np.array(stddev).flatten()
    return stddev

def contrast_entropy(img):
    contr_entr_features = []
    stds = StdBGR(img)
    entrs = Entropy(img)
    Add = stds + entrs
    Mult = stds * entrs
    contr_entr_features.append(entrs)
    contr_entr_features.append(stds)
    contr_entr_features.append(Add)
    contr_entr_features.append(Mult)   
    return np.array(contr_entr_features).flatten()

def calculate_kurtosis(img):
    B, G, R = cv2.split(img)

    kurtosis_B = kurtosis(B.ravel(), fisher=True)
    kurtosis_G = kurtosis(G.ravel(), fisher=True)
    kurtosis_R = kurtosis(R.ravel(), fisher=True)

    kurtosis_features = [kurtosis_B, kurtosis_G, kurtosis_R]
    
    return kurtosis_features 

def calculate_lbp(image, scales=[1.0, 0.5, 0.25], P=8, R=1):
    gray_image = rgb2gray(image)
    gray_image = img_as_ubyte(gray_image)
    multi_res_hist = []

    for scale in scales:
        scaled_image = rescale(gray_image, scale, anti_aliasing=True, mode='reflect')
        scaled_image = img_as_ubyte(scaled_image)
        lbp = local_binary_pattern(scaled_image, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P * (P - 1) + 3), density=True)
        hist /= (hist.sum() + 1e-6)
        multi_res_hist.extend(hist)

    return np.array(multi_res_hist)

def calculate_hog(image):
    image=rgb2gray(image)
    image = np.asarray(image)
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=False)
    return np.array(features)

def calculate_glcm(images):
    image = (images[0] * 255).astype(np.uint8)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    for angle in angles:
        glcm = graycomatrix(image, distances, [angle], symmetric=True, normed=True)
        for prop in props:
            feature = graycoprops(glcm, prop)
            features.extend(feature.flatten())
    
    return np.array(features)

def calculate_gabor(image):
    image = np.asarray(image)
    frequencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    image_features = []
    for frequency in frequencies:
        filt_real, filt_imag = gabor(image[0], frequency)
        image_features.append(filt_real.mean())
        image_features.append(filt_real.var())
        image_features.append(filt_imag.mean())
        image_features.append(filt_imag.var())
    return np.array(image_features)

def histogram_color_features(image):
    # Split the image into its three color channels (BGR in OpenCV)
    blue_channel, green_channel, red_channel = cv2.split(image)
    
    # Define the number of bins for the histograms
    bins = 256
    
    # Compute histograms for each channel
    blue_hist = cv2.calcHist([blue_channel], [0], None, [bins], [0, 256]).flatten()
    green_hist = cv2.calcHist([green_channel], [0], None, [bins], [0, 256]).flatten()
    red_hist = cv2.calcHist([red_channel], [0], None, [bins], [0, 256]).flatten()

    # Normalize histograms
    blue_hist /= np.sum(blue_hist)
    green_hist /= np.sum(green_hist)
    red_hist /= np.sum(red_hist)
    
    # Calculate statistical measures for each histogram
    def histogram_stats(hist):
        peak_position = np.argmax(hist)  # Position of the peak
        bin_count = hist[peak_position]  # Count at the peak position
        mean = np.mean(hist)
        var = np.var(hist)
        skewness = skew(hist)
        kurt = kurtosis(hist)
        return [peak_position, bin_count, mean, var, skewness, kurt]
    
    # Extract features from each channel's histogram
    blue_features = histogram_stats(blue_hist)
    green_features = histogram_stats(green_hist)
    red_features = histogram_stats(red_hist)
    
    # Concatenate all features into a single feature vector
    feature_vector = np.array(blue_features + green_features + red_features).flatten()
    
    return feature_vector

def dft_features(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform the Discrete Fourier Transform
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center
    
    # Extract magnitude and phase
    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)
    
    # Calculate statistical features from the magnitude and phase
    magnitude_mean = np.mean(magnitude)
    magnitude_variance = np.var(magnitude)
    phase_mean = np.mean(phase)
    phase_variance = np.var(phase)
    
    return np.array([magnitude_mean, magnitude_variance, phase_mean, phase_variance])


def radon_features(image):

    angles=np.linspace(0., 180., max(image.shape), endpoint=False)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute Radon transform for a set of angles
    sinogram = radon(gray_image, theta=angles, circle=True)
    
    # Extract statistical features from the Radon transform (sinogram)
    radon_mean = np.mean(sinogram)
    radon_variance = np.var(sinogram)
    radon_skewness = skew(sinogram.ravel())
    radon_kurtosis = kurtosis(sinogram.ravel())
    
    return np.array([radon_mean, radon_variance, radon_skewness, radon_kurtosis])

def wavelet_features(image, wavelet='haar'):
    # Split the image into color channels
    blue_channel, green_channel, red_channel = cv2.split(image)
    
    def extract_wavelet_coeffs(channel):
        # Perform a single level 2D Discrete Wavelet Transform
        coeffs = pywt.dwt2(channel, wavelet)
        cA, (cH, cV, cD) = coeffs  # Approximation, Horizontal, Vertical, Diagonal

        # Calculate statistics for each set of coefficients
        features = []
        for coef in [cA, cH, cV, cD]:
            features.append(np.mean(coef))
            features.append(np.var(coef))
            features.append(skew(coef.ravel()))
            features.append(kurtosis(coef.ravel()))
        return features

    # Extract features from each color channel
    blue_features = extract_wavelet_coeffs(blue_channel)
    green_features = extract_wavelet_coeffs(green_channel)
    red_features = extract_wavelet_coeffs(red_channel)
    
    # Combine features from all channels
    wavelet_feature_vector = np.array(blue_features + green_features + red_features).flatten()
    
    return wavelet_feature_vector

def haralick_features(image):

    # Convert each color channel to grayscale and compute GLCM for each
    channels = cv2.split(image)
    distances = [1]  # Distance for GLCM computation
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Common angles in radians
    
    haralick_features_vector = []
    
    for channel in channels:
        # Ensure the channel is in uint8 format for GLCM calculation
        channel = channel.astype(np.uint8)
        
        # Compute the GLCM
        glcm = graycomatrix(channel, distances=distances, angles=angles, symmetric=True, normed=True)
        
        # Compute Haralick properties
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        
        # Append features for this channel
        haralick_features_vector.extend([contrast, correlation, energy, homogeneity, dissimilarity])
    
    # Return the combined feature vector for all channels
    return np.array(haralick_features_vector)

def more_features(image):
    w = wavelet_features(image)
    r = radon_features(image)
    d = dft_features(image)
    h = histogram_color_features(image)
    haralick = haralick_features(image)
    return np.concatenate([w, r, d, h, haralick])

def shape_features(binary_image):
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    binary_image = binary_image.astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(8)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    rect_area = w * h
    extent = area / float(rect_area + 1e-6)  # Avoid division by zero
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area + 1e-6)
    _, (MA, ma), angle = cv2.fitEllipse(cnt) if len(cnt) >= 5 else (0, (0, 0), 0)

    return np.array([area, perimeter, aspect_ratio, extent, solidity, MA, ma, angle])

def boundary_features(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(4)

    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    convex_perimeter = cv2.arcLength(hull, True)
    fractal_dimension = np.log(perimeter + 1e-6) / np.log(convex_perimeter + 1e-6)
    sobel_x = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(binary_image, cv2.CV_64F, 0, 1, ksize=5)
    edge_gradient_magnitude = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

    return np.array([perimeter, convex_perimeter, fractal_dimension, edge_gradient_magnitude])

def asymmetry_orientation_features(binary_image):
    labeled_image = label(binary_image)
    props = regionprops(labeled_image)
    if not props:
        return np.zeros(3)

    orientation = props[0].orientation
    centroid = props[0].centroid
    center_x, center_y = int(centroid[1]), int(centroid[0])
    half_top, half_bottom = binary_image[:center_y, :], binary_image[center_y:, :]
    half_left, half_right = binary_image[:, :center_x], binary_image[:, center_x:]
    total_sum = np.sum(binary_image) + 1e-6  # Avoid division by zero

    asymmetry_horizontal = abs(np.sum(half_top) - np.sum(half_bottom)) / total_sum
    asymmetry_vertical = abs(np.sum(half_left) - np.sum(half_right)) / total_sum

    return np.array([orientation, asymmetry_horizontal, asymmetry_vertical])


def hausdorff_feature(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0])

    cnt = contours[0]
    contour_points = cnt.reshape(-1, 2)
    if len(contour_points) < 2:
        return np.array([0])

    hausdorff_dist = directed_hausdorff(contour_points, contour_points)[0]

    return np.array([hausdorff_dist])

def segmentation_features(binary_image):
    shape_feats = shape_features(binary_image)
    boundary_feats = boundary_features(binary_image)
    asymmetry_orientation_feats = asymmetry_orientation_features(binary_image)
    hausdorff_feat = hausdorff_feature(binary_image)

    return np.concatenate([shape_feats, boundary_feats, asymmetry_orientation_feats, hausdorff_feat])


# def GLCM(image):
#     gray_image = rgb2gray(image)  # Converts to range [0, 1]
#     gray_image = (gray_image * 255).astype(int)
#     levels = 256  # You can change this based on your requirement
#     glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=levels, symmetric=True, normed=True)
#     return glcm

# def LBP(image):
#     gray_image = rgb2gray(image)
#     gray_image = (gray_image * 255).astype(int)
#     LBP=local_binary_pattern(gray_image,method="ror",R=3, P=24)
#     return LBP

# def Gabor(image,frequency):
#     gray_image = rgb2gray(image)
#     real_response, imag_response = filters.gabor(gray_image, frequency=frequency)
#     gabor_real_mean = np.mean(real_response)
#     gabor_real_variance = np.var(real_response)
#     gabor_imag_mean = np.mean(imag_response)
#     gabor_imag_variance = np.var(imag_response)
#     gabor_features = [gabor_real_mean, gabor_real_variance, gabor_imag_mean, gabor_imag_variance]
#     return gabor_features
