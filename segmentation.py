from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import cv2

np.random.seed(42)


clusters_list = []
cluster = {}
centers = {}

class meanshift():
    def __init__(self, img,iter):
        self.result_image = np.zeros(img.shape,np.uint8)   #Initializes an output image (result_image) with the same size and data type
        self.radius = 90            #spatial radius set to 90. This represents the maximum distance a pixel can be from the seed point to be considered a neighbor.
        self.Iter = iter

    def getNeighbors(self,seed,matrix):
        """

        -This function takes a seed point (seed) and a feature matrix (matrix) as input.
        -This function essentially identifies pixels within the spatial radius of the seed point.
        -It iterates through each row in the matrix (representing a pixel)
         and calculates the Euclidean distance between the seed and that pixel.

        """
        neighbors = []
        for i in range(0,len(matrix)):
            Pixel = matrix[i]
            d = math.sqrt(sum((Pixel-seed)**2))
            if(d<self.radius):      #If the distance is less than the radius, the pixel's index is added to the neighbors list.
                 neighbors.append(i)
        return neighbors

    def markPixels(self,neighbors,mean,matrix):
        """

        -This function takes neighbors (neighbors), mean (mean), feature matrix (matrix), and cluster number (cluster) as input.
        -It iterates through the neighbors list (indices of pixels within spatial radius).
        -For each neighbor, it updates the corresponding location in the output image (result_image) with the mean value
        """
        for i in neighbors:
            Pixel = matrix[i]
            x=Pixel[3]
            y=Pixel[4]
            self.result_image[x][y] = np.array(mean[:3],np.uint8)
        return np.delete(matrix,neighbors,axis=0)   # removes the processed neighbors from the feature matrix (matrix)

    def calculateMean(self,neighbors,matrix):
        """

        -This function takes neighbors (neighbors) and feature matrix (matrix) as input.
        -It selects the corresponding rows from the feature matrix based on the provided neighbor indices.
        -It calculates the average for each feature (Red, Green, Blue, X-coordinate, Y-coordinate) using np.mean.
        -This function essentially computes the new mean (center) based on the neighboring pixels.

        """
        neighbors = matrix[neighbors]       #selects rows from the matrix based on the indices in the neighbors list.
        r=neighbors[:,:1]
        g=neighbors[:,1:2]
        b=neighbors[:,2:3]
        x=neighbors[:,3:4]
        y=neighbors[:,4:5]
        mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])

        return mean

    def createFeatureMatrix(self,img):
        """
        -This function takes an image (img) as input.
        -It iterates through each pixel in the image and creates a feature vector containing the pixel's color values (Red, Green, Blue) and its spatial coordinates (X, Y).
        -It converts the list of feature vectors into a NumPy array (F) for efficient processing.
        """
        h,w,d = img.shape
        F = []
        for row in range(0,h):
            for col in range(0,w):
                r,g,b = img[row][col]
                F.append([r,g,b,row,col])
        F = np.array(F)
        return F

    def performMeanShift(self,img):
        """

        This function performs the core mean shift segmentation process

        """

        F = self.createFeatureMatrix(img)   #creates a feature matrix
        while(len(F) > 0):
            randomIndex = randint(0,len(F)-1)       # It randomly selects an index from the feature matrix as the initial seed point
            seed = F[randomIndex]       #retrieves the feature vector (color and coordinates)
            initialMean = seed
            neighbors = self.getNeighbors(seed,F)   # finds neighboring pixels

            if(len(neighbors) == 1):
                F=self.markPixels([randomIndex],initialMean,F)      #if single pixel mark directly to the output and skip the iteration
                continue
            mean = self.calculateMean(neighbors,F)
            meanShift = abs(mean-initialMean)

            if(np.mean(meanShift)<self.Iter):
                F = self.markPixels(neighbors,mean,F)

        return self.result_image



def kmeans(img, max_iter=100, K=2, threshold=0.85):
    """
    Apply K-Means clustering to an image to segment it into K colors.

    Parameters:
    - img: Input image (numpy array)
    - max_iter: Maximum number of iterations for K-Means algorithm (default: 100)
    - K: Number of clusters (default: 2)
    - threshold: Convergence threshold for centroid updates (default: 0.85)

    Returns:
    - segmented_image: Segmented image after applying K-Means clustering
    """



    # Change color to RGB (from BGR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flatten the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))

    # Convert pixel values to float type
    pixel_vals = np.float32(pixel_vals)

    # Handle NaN values in pixel values array by setting them to a small value
    pixel_vals[np.isnan(pixel_vals)] = 1e-6

    # Initialize cluster centroids randomly
    centroids = pixel_vals[np.random.choice(pixel_vals.shape[0], K, replace=False), :]

    # Handle NaN values in centroids array by setting them to a small value
    centroids[np.isnan(centroids)] = 1e-6

    # Initialize old centroids
    old_centroids = np.zeros_like(centroids)

    # Iteratively update centroids until convergence or maximum iterations reached
    for i in range(max_iter):
        # Assign each pixel to the nearest centroid
        distances = np.sqrt(((pixel_vals - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update cluster centroids
        for k in range(K):
            centroids[k] = np.mean(pixel_vals[labels == k], axis=0)

        # Handle NaN values in centroids array by setting them to a small value
        centroids[np.isnan(centroids)] = 1e-6

        # Check for convergence
        if np.abs(centroids - old_centroids).mean() < threshold:
            break
        old_centroids = centroids.copy()

    # Convert centroids to 8-bit values
    centers = np.uint8(centroids)

    # Assign each pixel to its corresponding cluster center
    segmented_data = centers[labels.flatten()]

    # Reshape segmented data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image


def region_growing(img, seed_point, threshold):
    """
    Apply region growing algorithm to an image from a seed point.

    Parameters:
    - img: Input image (numpy array)
    - seed_point: Seed point (tuple of x, y coordinates)
    - threshold: Threshold for similarity between pixels (float)

    Returns:
    - output_img: Image with region grown from the seed point
    """

    # Make a copy of the input image
    output_img = img.copy()

    # Get the dimensions of the image
    height, width, channels = img.shape

    # Initialize an empty image for output
    output = np.zeros_like(img, dtype=np.uint8)

    # Function to get neighboring pixels of a given point
    def get_neighbours(point):
        x, y = point
        neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbours if 0 <= n[0] < height and 0 <= n[1] < width]

    # Function to check similarity between two pixels
    def similarity(pixel1, pixel2, threshold):
        return np.sqrt(np.sum((pixel1 - pixel2)**2)) < threshold

    # Initialize a queue with the seed point
    queue = [seed_point]

    # Start region growing
    while queue:
        current_point = queue.pop(0)
        output[current_point] = (0, 255, 0)  # Set the pixel to green
        neighbours = get_neighbours(current_point)
        for neighbour in neighbours:
            # Check if the neighbor is not already visited and similar to the seed value
            if not np.any(output[neighbour]) and similarity(img[neighbour], img[seed_point], threshold):
                output[neighbour] = (0, 255, 0)  # Set the pixel to green
                queue.append(neighbour)

    # Create a mask for the green regions
    green_mask = np.all(output == (0, 255, 0), axis=-1)

    # Overlay the green regions onto the original image
    output_img[green_mask] = (0, 255, 0)

    return output_img





def  Euclidean_distance(x1, x2):
    """
   This function calculates the Euclidean distance between two data points x1 and x2
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def clusters_average_distance(cluster1, cluster2):

    """
        1- calculates the average distance between two clusters.
        2- calculates the center of each cluster by taking the average of all data points within that cluster.
        3- calls the previously defined Euclidean_distance function to find the distance between the two cluster centers.
    """
    cluster1_center = np.average(cluster1)
    cluster2_center = np.average(cluster2)
    return  Euclidean_distance(cluster1_center, cluster2_center)


def initial_clusters(image_clusters):
    """

   Performs initial clustering of data points (pixels) into initial_k  clusters.

    """
    global initial_k    #define the initial number of clusters
    groups = {}         #Dictionary to store data points belonging to each cluster, identified by their average color.
    cluster_color = int(256 / initial_k)
    for i in range(initial_k):
        color = i * cluster_color
        groups[(color, color, color)] = []
    for i, p in enumerate(image_clusters):
        Cluster = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))    #finds the closest cluster center (represented by average color) in the groups dictionary
                                                                                # using min and a distance function.
        groups[Cluster].append(p)            #Data points are assigned to their closest clusters
    return [group for group in groups.values() if len(group) > 0]   #filter empty clusters


def get_cluster_center(point):
    """
   retrieves the cluster center for a given data point (point)
    """
    point_cluster_num = cluster[tuple(point)]   #Uses the cluster dictionary to find the cluster number associated with the data point
    center = centers[point_cluster_num]         #Retrieves the cluster center from the centers dictionary based on the cluster number.
    return center


def get_clusters(image_clusters):
    """
     This function performs the core agglomerative clustering process.
    """
    clusters_list = initial_clusters(image_clusters)

    while len(clusters_list) > clusters_number:
        cluster1, cluster2 = min(
            [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
            key=lambda c: clusters_average_distance(c[0], c[1]))        # find the pair of clusters (represented as cluster1 and cluster2) that are most similar based on their average distance.

        clusters_list = [cluster_itr for cluster_itr in clusters_list if
                         cluster_itr != cluster1 and cluster_itr != cluster2]   # filtered to remove the two clusters that were merged (cluster1 and cluster2).

        merged_cluster = cluster1 + cluster2

        clusters_list.append(merged_cluster)

    for cl_num, cl in enumerate(clusters_list): #assigning cluster num to each point
        for point in cl:
            cluster[tuple(point)] = cl_num

    for cl_num, cl in enumerate(clusters_list): #Computing cluster centers
        centers[cl_num] = np.average(cl, axis=0)


def apply_agglomerative_clustering( number_of_clusters, initial_number_of_clusters, image):
    global clusters_number
    global initial_k

    resized_image = cv2.resize(image, (256, 256))

    clusters_number = number_of_clusters
    initial_k = initial_number_of_clusters
    flattened_image = np.copy(resized_image.reshape((-1, 3)))

    get_clusters(flattened_image)
    output_image = []
    for row in resized_image:
        rows = []
        for col in row:
            rows.append(get_cluster_center(list(col)))
        output_image.append(rows)
    output_image = np.array(output_image, np.uint8)

    return output_image








