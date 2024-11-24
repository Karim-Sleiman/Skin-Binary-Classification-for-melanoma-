from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    MapTransform,
    ToNumpyd,
    ToTensord,
    ScaleIntensityd,
    ToTensord
    )
import numpy as np
import cv2
import torch
from monai.transforms import MapTransform
from sklearn.cluster import KMeans
from skimage.color import rgb2gray


import numpy as np
import cv2
from skimage.color import rgb2gray
from monai.transforms import MapTransform
from scipy.ndimage import label

class Otsud(MapTransform): 
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        
    def otsu_thresholding(self, input):
        if input.shape[0] != 3:  # Ensure the input is a 3-channel image
            raise ValueError("Input must be an RGB image with 3 channels.")
        
        input_rearranged = np.transpose(input, (1, 2, 0))
        gray_image = rgb2gray(input_rearranged)  # Ensure gray_image is float (0-1)
        gray_image = (gray_image * 255).astype(np.uint8)  # Convert to uint8 for cv2.threshold
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = np.expand_dims(thresh, 0)  # Adjust shape if needed
        thresh = 255 - thresh  # Invert the thresholded image
        
        # Remove edge-connected components
        thresh = self.remove_edge_connected_components(thresh[0])  # Process 2D binary image
        return thresh[np.newaxis, :]  # Return with the correct shape

    def remove_edge_connected_components(self, binary_image):
        # Create a copy of the binary image for processing
        labeled_image, num_labels = label(binary_image)
        
        # Create a mask for components that touch the edges
        edge_connected_mask = np.zeros_like(binary_image, dtype=bool)
        
        # Check the first and last rows and columns for edge-connected components
        edge_connected_mask[0, :] = 1
        edge_connected_mask[-1, :] = 1
        edge_connected_mask[:, 0] = 1
        edge_connected_mask[:, -1] = 1
        
        # Label the edge-connected components
        edge_connected_labels = np.unique(labeled_image[edge_connected_mask])
        
        # Remove all edge-connected components from the binary image
        for label_value in edge_connected_labels:
            if label_value > 0:  # Exclude the background label (0)
                binary_image[labeled_image == label_value] = 0
                
        return binary_image
    
    def __call__(self, data):
        for key in self.key_iterator(data):
            output_key = self.output_key if self.output_key is not None else key
            data[output_key] = self.otsu_thresholding(data[key])
        return data

class SegmentUsingOtsu(MapTransform):
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def segment_image(self, original_image, otsu_image):
        # Convert Otsu image to shape (256, 256)
        image_otsu = otsu_image.squeeze()  # Remove singleton dimensions if any
        # Convert original image to shape (256, 256, 3)
        image = original_image.numpy().transpose(1, 2, 0)

        # Create a mask where Otsu image is 255
        mask = image_otsu == 255  # mask will be (256, 256)

        # Prepare the multiplied image
        multiplied_image = np.zeros_like(image)  # Initialize as a zero array of the same shape as image

        # Use the mask to set the corresponding pixels in multiplied_image
        multiplied_image[mask] = image[mask]  # Assign pixel values where the mask is True

        return torch.from_numpy(multiplied_image)  # Convert back to tensor

    def __call__(self, data):
        for key in self.key_iterator(data):
            output_key = self.output_key or key
            original_image = data[key]  # Original image tensor
            otsu_image = data['image_otsu']  # Otsu segmented image tensor
            
            data[output_key] = self.segment_image(original_image, otsu_image)
        return data



class PositiveExtractiond(MapTransform):
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        
    def positive(self, input):
        positive_points = np.where(input == 255)  # Extracting points where intensity is 255 (white regions)
        return positive_points
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.positive(d[key])
        return d

    
    
class NegativeExtractiond(MapTransform):
    def __init__(self, keys, num_clusters=2, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.num_clusters = num_clusters
        self.output_key = output_key
        
    def negative(self, input):
        return (255-input)/255
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.negative(d[key])
        return d
    
class IterativeWatershedd(MapTransform):
    def __init__(self, keys, positive_key, negative_key, iterations=1, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.iterations = iterations
        
    def watershed(self, image, pos_marker, neg_marker, iterations):
        im = np.mean(image, axis=0).astype(np.uint8)
        markers = np.zeros_like(im, dtype=int)
        markers[pos_marker > 0] = 1
        markers[neg_marker > 0] = 2
        labels = cv2.watershed(np.stack([im, im, im], -1), markers)
        
        for _ in range(iterations):
            markers = np.zeros_like(markers)
            markers[labels == 1] = 1
            markers[labels == 2] = 2
            labels = cv2.watershed(np.stack([im, im, im], -1), markers)
        
        labels[labels == -1] = 0
        labels = np.expand_dims(labels.astype(np.uint8), 0)
        return labels
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.watershed(
                d[key], d[self.positive_key], d[self.negative_key], self.iterations
            )
        return d


class MorphologicalErosiond(MapTransform):
    def __init__(self, keys, kernel_size=(3, 3), iterations=1, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.output_key = output_key
        
    def perform_erosion(self, input):
        kernel = np.ones(self.kernel_size, np.uint8)
        erosion = cv2.erode(input, kernel, iterations=self.iterations)
        erosion = np.expand_dims(erosion, 0)
        return erosion
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.perform_erosion(d[key][0])
        return d


class MorphologicalDilationd(MapTransform):
    def __init__(self, keys, kernel_size=(3, 3), iterations=1, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.output_key = output_key
        
    def perform_dilation(self, input):
        kernel = np.ones(self.kernel_size, np.uint8)
        dilation = cv2.dilate(input, kernel, iterations=self.iterations)
        dilation = np.expand_dims(dilation, 0)
        return dilation
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.perform_dilation(d[key][0])
        return d


class BlackEdgeRemovald(MapTransform):
    def __init__(self, keys, output_key="edge_removed", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def remove_black_edges(self, input_image):
        processed_image = input_image.clone()
        center_y, center_x = processed_image.shape[1] // 2, processed_image.shape[2] // 2
        distance_threshold = 90
        y_indices, x_indices = np.ogrid[:processed_image.shape[1], :processed_image.shape[2]]
        distance_from_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        distance_mask = distance_from_center >= distance_threshold
        combined_mask = (processed_image < 0.03).any(axis=0) & distance_mask

        for c in range(processed_image.shape[0]):
            processed_image[c][combined_mask] = processed_image[c].max()

        processed_image_np = processed_image.numpy()
        kernel = np.ones((7, 7), np.uint8)

        for c in range(processed_image_np.shape[0]):
            processed_image_np[c] = cv2.morphologyEx(processed_image_np[c], cv2.MORPH_OPEN, kernel)

        processed_image = torch.from_numpy(processed_image_np)
        return processed_image

    def __call__(self, data):
        for key in self.key_iterator(data):
            output_key = self.output_key
            data[output_key] = self.remove_black_edges(data[key])
        return data


class KMeansSegmentd(MapTransform):
    def __init__(self, keys, num_clusters=2, output_key='kmeans', mask_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.num_clusters = num_clusters
        self.output_key = output_key
        self.mask_key = mask_key
    
    def segment(self, input_image, mask=None):
        if mask is not None:
            input_image = input_image * mask
        pixels = input_image.numpy().reshape(-1, input_image.shape[0]).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, self.num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(input_image.shape[1], input_image.shape[2], input_image.shape[0])
        dominant_cluster_mask = (segmented_image == segmented_image.max(axis=2, keepdims=True)).astype(np.uint8)
        return torch.from_numpy(np.moveaxis(dominant_cluster_mask, -1, 0))

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.segment(d[key], mask=d.get(self.mask_key))
        return d

class HairRemoval(MapTransform):
    def _init_(self, keys, output_key=None, allow_missing_keys=False, kernel_size=15):
        super()._init_(keys, allow_missing_keys)
        self.output_key = output_key
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def apply_blackhat_subtraction(self, input_image):
        processed_image_np = input_image.clone().numpy()
        for c in range(processed_image_np.shape[0]):
            blackhat = cv2.morphologyEx(processed_image_np[c], cv2.MORPH_BLACKHAT, self.kernel)
            processed_image_np[c] = processed_image_np[c] - blackhat
        processed_image = torch.from_numpy(processed_image_np)
        return processed_image

    def _call_(self, data):
        for key in self.key_iterator(data):
            output_key = self.output_key or key
            data[output_key] = self.apply_blackhat_subtraction(data[key])
        return data


transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),  # Load images
    Resized(keys=['image'], spatial_size=(256,256)),  # Resize all images to 256x256
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image']),
    # BlackEdgeRemovald(keys=["image"], output_key='edge_removed'),
    Otsud(keys=["image"], output_key='image_otsu'),
    # SegmentUsingOtsu(keys=['image'], output_key='segmented_image'),  # Segment using Otsu result
    # PositiveExtractiond(keys=["image_otsu"], output_key='pos_markers'),
    # NegativeExtractiond(keys=["image_otsu"], output_key='neg_markers'),
    # IterativeWatershedd(keys='image', iterations=5, positive_key='pos_markers', negative_key='neg_markers', output_key='watershed'),  
    # KMeansSegmentd(keys='clahe_kmeans', num_clusters=2, output_key='kmeans'),
    # HairRemoval(keys=['image'])
    # # ScaleIntensityd(keys=['edgeremoved']),
    # KMeansSegmentd(keys=["edgeremoved"], num_clusters=2, output_key='kmeans')  # Change from list to string
])
