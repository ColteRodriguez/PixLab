import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.extmath import randomized_svd
import sklearn.preprocessing as preprocessing
import os
from PIL import Image
import cv2
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
from colorama import Fore

"""
KMask is a clustering model framework which has properties:

im: an input image to classify

"""
class KMask:
    def __init__(self, im):
        self.im = im
        self.shape = None
        self.gt = None
        self.PCs = None
        self.k = None
        self.f = None
        self.percentages = None
        self.accuracy, self.iou, self.SSD, self.predicted_labels, self.kmeans = None, None, None, None, None
        self.sampledatapaths = dataset = {"64R-18W-01-20":
           {"full": 
            ["64R-18W-01-20/full/64R-18W-01-20_1.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_2.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_3.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_4.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_5.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_6.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_7.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_8.tif", 
             "64R-18W-01-20/full/64R-18W-01-20_9.tif"], 
            "partial": 
            ["64R-18W-01-20/partial/64R-18W-01-20_1.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_2.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_3.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_4.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_5.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_6.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_7.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_8.tif", 
             "64R-18W-01-20/partial/64R-18W-01-20_9.tif"],
            "RGB": 
            ["64R-18W-01-20/RGB/Img0008.tif"],
            "ground_truth":
            "64R-18W-01-20/ground_truth/64R-18W-01-20_9.png"},
          "65R-01W-15-17":
          {"full": 
            ["65R-01W-15-17/full/65R-01W-15-17_1.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_2.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_3.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_4.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_5.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_6.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_7.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_8.tif", 
             "65R-01W-15-17/full/65R-01W-15-17_9.tif"],            
            "partial": 
            ["65R-01W-15-17/partial/65R-01W-15-17_1.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_2.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_3.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_4.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_5.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_6.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_7.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_8.tif", 
             "65R-01W-15-17/partial/65R-01W-15-17_9.tif"],
           "RGB":
           ["65R-01W-15-17/RGB/Img0016c.tif"],
          "ground_truth":
          "65R-01W-15-17/ground_truth/65R-01W-15-17.png"},
          "88R-01W-40-41":
          {"full": 
            ["88R-01W-40-41/full/88R-01W-40-41_1.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_2.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_3.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_4.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_5.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_6.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_7.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_8.tif", 
             "88R-01W-40-41/full/88R-01W-40-41_9.tif"],            
            "partial": 
            ["88R-01W-40-41/partial/88R-01W-40-41_1.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_2.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_3.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_4.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_5.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_6.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_7.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_8.tif", 
             "88R-01W-40-41/partial/88R-01W-40-41_9.tif"],
           "RGB":
           ["88R-01W-40-41/RGB/Img0086c.tif"],
          "ground_truth":
          "88R-01W-40-41/ground_truth/88R-01W-40-41.png"}}

    '''
    Combines input multispectral images into a tensor of shape (n,m,p), where p in the number of bands for the image.
    
    Input:
    - Bands: An array of paths to images. If the array has length one, the image is assumed to be grayscale or RGB. If the array has length > 1, the image is assumed to be mutispectral and a composition is constructed.
    
    Output: 
    - A composite image and its shape (n,m,p)
    '''
    def generate_multispec_image(self, bands):
        if len(bands) == 1:
            return cv2.imread(bands[0]), cv2.imread(bands[0]).shape 


        red_band_images = []
        # Process each image
        for path in bands:
            # Load the image
            img = Image.open(path)

            # Convert to numpy array and select the red channel
            img_array = np.array(img)
            red_band = img_array[:, :, 0]  # Select only the red channel

            # Reshape to (100, 100, 1) and add to the list
            red_band_images.append(red_band[:, :, np.newaxis])

        # Stack along the last axis to get a shape of (100, 100, 9)
        image = np.concatenate(red_band_images, axis=2)
        # print(f"multispec generation sucessful: found {image.shape[2]} bands for image with shape {image.shape}")
        return image, image.shape
      
    '''
    Returns an image reshaped from (n,m,p) to (n*m,p).
    '''
    def unfold(self, image):
        shape = image.shape
        return np.reshape(image, (shape[0]*shape[1], shape[2]))

    '''
    Returns an image reshaped from (n*m,p) to (n,m,p).
    '''
    def fold(self, image, shape):
        return np.reshape(image, (shape[0], shape[1], ))
    
    '''
    Generates an image composition of imput Principal components, which is an
    array of length 1 to p where p is the number of image bands (n,m,p)
    output is n,m,(# principal components)
    '''
    def PCA(self, image, PCs):
        self.PCs = PCs
        num_components = image.shape[2]
        unfolded_im = self.unfold(image)
        pca = PCA(n_components=num_components)
        reduced_points = pca.fit_transform(unfolded_im)[:, PCs]
        return reduced_points.reshape(self.shape[0], self.shape[1], len(self.PCs))

    '''
    Just SKlearn kmeans returns the model params and the labels
    '''
    def k_means_clustering(self, v, k):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(v)
        return kmeans.labels_.reshape((self.shape[0], self.shape[1])), kmeans
     
    '''
    Returns labels for ground truth image. This is just a simple kmeans because the ground truth 
    image is already labeled, this is just the computer interpreting the man-made labels
    '''
    def get_ground_truth(self, gt):
        im = cv2.resize(cv2.imread(gt), (self.shape[1], self.shape[0], ))
        image, shape = im, im.shape
        _, labels = self.k_means_clustering(self.unfold(im), self.k)
        gmask = np.reshape(labels, (self.shape[0], self.shape[1]))
        return gmask
    
    '''
    iou score for comparing two images
    '''
    def iou_score(self, true_labels, predicted_labels, k):
        iou = []
        for label in range(k):
            intersection = np.logical_and(true_labels == label, predicted_labels == label).sum()
            union = np.logical_or(true_labels == label, predicted_labels == label).sum()
            if union == 0:  # Avoid division by zero
                iou.append(1 if intersection == 0 else 0)  # Perfect score if both empty
            else:
                iou.append(intersection / union)
        return np.mean(iou)
    
    '''
    Formula for compositional accuracy
    '''
    def calculate_percentage_accuracy(image1, image2):
        # Helper function to get value percentages for an image
        def get_value_percentages(image):
            unique, counts = np.unique(image, return_counts=True)
            total_pixels = image.size
            return {value: (count / total_pixels) * 100 for value, count in zip(unique, counts)}

        # Get value percentages for both images
        percentages1 = get_value_percentages(image1)
        percentages2 = get_value_percentages(image2)

        # Get all unique values in both images
        all_values = set(percentages1.keys()).union(percentages2.keys())

        # Calculate mean absolute error between percentages
        total_error = 0
        for value in all_values:
            # Use 0 if a value is not present in an image
            percent1 = percentages1.get(value, 0)
            percent2 = percentages2.get(value, 0)
            total_error += abs(percent1 - percent2)

        # Calculate average error
        mean_absolute_error = total_error / len(all_values)
        accuracy_score = 100 - mean_absolute_error  # Convert to an "accuracy" score

        return accuracy_score

    '''
    Leueng, Malik (LM) applied filter bank 
    '''
    def build_filter_predictor(self, im, km):
        # Building some gabor kernels to filter image
        orientations = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        wavelengths = [3, 6, 12, 24, 48, 96]

        def build_gabor_kernels():
            filters = []
            ksize = 40
            for rotation in orientations:
                for wavelength in wavelengths:
                    kernel = cv2.getGaborKernel((ksize, ksize), 4.25, rotation, wavelength, 3, 0, ktype=cv2.CV_32F)
                    filters.append(kernel)

            return filters

        image, shape = im, im.shape
        rows, cols, channels = image.shape
        # Resizing the image. 
        # Full image is taking to much time to process
        # image = cv2.resize(image, (int(cols * 0.5), int(rows * 0.5))).reshape((int(rows * 0.5), int(cols * 0.5), channels))
        # rows, cols, channels = image.shape

        gray = np.mean(image, axis=2)

        gaborKernels = build_gabor_kernels()

        gaborFilters = []

        for (i, kernel) in enumerate(gaborKernels):
            filteredImage = cv2.filter2D(gray, cv2.CV_8UC1, kernel)

            # Blurring the image
            sigma = int(3*0.5*wavelengths[i % len(wavelengths)])

            # Sigma needs to be odd
            if sigma % 2 == 0:
                sigma = sigma + 1

            blurredImage = cv2.GaussianBlur(filteredImage,(int(sigma),int(sigma)),0)
            gaborFilters.append(blurredImage)


        # numberOfFeatures = 1 (gray color) + number of gabor filters + 2 (x and y)
        numberOfFeatures = 1  + len(gaborKernels) + 2

        # Empty array that will contain all feature vectors
        featureVectors = []

        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                vector = [gray[i][j]]

                for k in range(0, len(gaborKernels)):
                    vector.append(gaborFilters[k][i][j])

                vector.extend([i+1, j+1])

                featureVectors.append(vector)

        # Normalizing the feature vectors
        scaler = preprocessing.StandardScaler()

        scaler.fit(featureVectors)
        featureVectors = scaler.transform(featureVectors)

        kmeans = KMeans(n_clusters=km, n_init=10)
        kmeans.fit(featureVectors)

        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        return labels.reshape(rows, cols, 1)

    '''
    Percentages of dominant lithologies in the sample
    '''
    def percentages_(self):
        area = self.predicted_labels.size
        percentages = {}
        for label in range(self.k):
            # Count pixels with the current label
            label_count = np.sum(self.predicted_labels == label)

            # Compute percentage
            percentages[label] = (label_count / area) * 100
        # print(percentages)
        self.percentages = percentages
    
    
    '''
    *Not part of Rodriguez, 2025* Still in development so might not work quite right.
    '''
    def generate_image_composition(im: np.ndarray, bands: np.ndarray, func) -> np.ndarray:
        # Validate inputs
        if im.ndim != 3:
            raise ValueError("Input image 'im' must have shape (n, m, p).")

        n, m, p = im.shape

        if not (1 <= len(bands) <= p):
            raise ValueError("'bands' array length must be between 1 and p (inclusive).")

        if np.any((bands < 0) | (bands >= p)):
            raise ValueError("Band indices in 'bands' must be between 0 and p (exclusive).")

        # Select the specified bands
        selected_bands = [im[:, :, band] for band in bands]

        # Apply the user-defined function to generate the composition
        try:
            composed_image = func(*selected_bands)
        except Exception as e:
            raise ValueError(f"Error in applying the function: {e}")

        # Validate the output of the function
        if composed_image.shape != (n, m):
            raise ValueError("Output of 'func' must have shape (n, m).")

        return composed_image
    
    '''
    Run the prediction
    '''
    def predict(self, k, gt, transform):
        if k is not None:
            self.k = k                                                 # Set the model k property
        else:
            self.k = find_k()
            
        image, self.shape = self.generate_multispec_image(self.im)     # Get the original image data and set the shape propery
        
        # Pass the image through the preprocces pipeline defined by user. consists of filters, PCA compression, ...
        clustered_image = image
        for f, args, kwargs in transform:
            clustered_image = f(clustered_image, *args, **kwargs)

        # Run k-means with the same number of clusters as the labels
        predicted_labels, kmeans = self.k_means_clustering(self.unfold(clustered_image), k)
        
        if gt is not None:
            gt = gt[0]
            self.gt = self.get_ground_truth(gt)    # Get labels for GT data
            # Matching labels: Use the Hungarian algorithm to find the optimal assignment
            conf_matrix = confusion_matrix(self.gt.flatten(), predicted_labels.flatten())
            row_ind, col_ind = linear_sum_assignment(-conf_matrix)
            mapped_labels = np.zeros_like(predicted_labels)  # Remap predicted labels
            for i, j in zip(row_ind, col_ind):
                mapped_labels[predicted_labels == j] = i
            # Determine accuracy 
            accuracy = np.mean(mapped_labels == self.gt)
            num_labels = len(np.unique(self.gt))
            iou = self.iou_score(self.gt.flatten(), mapped_labels.flatten(), self.k)
            SSD = (kmeans.inertia_)/(self.shape[0]*self.shape[1])
            self.accuracy, self.iou, self.SSD, self.predicted_labels, self.kmeans = accuracy, iou, SSD, predicted_labels, kmeans
            self.percentages = self.percentages_()
            print(Fore.GREEN + f"Output Metrics: \nInput Image: {self.im}, shape={self.shape} \nPreprocessing: {transform} \nPCA dimension(s): {self.PCs} \nNumber of Facies Identified: {self.k} \nGround Truth?: {gt} \nAccuracy (not applicable if not ground truth passed): {self.accuracy} \nIoU (not applicable if not ground truth passed): {self.iou} \nSSD (model percision): {self.SSD}")
        else:
            SSD = (kmeans.inertia_)/(self.shape[0]*self.shape[1])
            # disp(cv2.imread(im[0]), predicted_labels)
            self.SSD, self.predicted_labels = SSD, predicted_labels
            self.percentages = self.percentages_()
            print(Fore.GREEN + f"Output Metrics: \nInput Image: {self.im}, shape={self.shape} \nPreprocessing: {transform} \nPCA dimension(s): {self.PCs} \nNumber of Facies Identified: {self.k} \nGround Truth?: {gt} \nAccuracy (not applicable if not ground truth passed): {self.accuracy} \nIoU (not applicable if not ground truth passed): {self.iou} \nSSD (model percision): {self.SSD}")

            
# Example of usage
def main():
    # These paths will mean nothing once they're off my deivce
    print("No Unit Tests")
    
if __name__ == "__main__":
    main()