import numpy as np
import os
# loading standard modules
import math
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.color import rgb2gray

class MyKmeansApp:

    def __init__(self, img, img_name, num_clusters=2, weightXY=1.0, dist_sensitve=0):
        self.k = num_clusters
        self.w = weightXY
        self.iteration = 0  # iteration counter
        self.SSE = np.infty  # SSE - "sum of squared errors" (SSE)
        self.dist_sensitve = dist_sensitve
        img_name = img_name.split('.')[0]
        self.img_name = img_name
        num_rows = self.num_rows = img.shape[0]
        num_cols = self.num_cols = img.shape[1]
        self.plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'plots_{img_name}_k={num_clusters}_dist_sensitive={dist_sensitve}')
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)
        self.im = img_as_ubyte(img)

        self.means = np.zeros((self.k, 5), 'd')  # creates a zero-valued (double) matrix of size Kx5
        self.__init_means()

        self.no_label = num_clusters  # special label value indicating pixels not in any cluster (e.g. not yet)

        # mask "labels" where pixels of each "region" will have a unique index-label (like 0,1,2,3,..,K-1)
        # the default mask value is "no-label" (K) implying pixels that do not belong to any region (yet)
        self.labels = np.full((num_rows, num_cols), fill_value=self.no_label, dtype=np.uint8)


    def __init_means(self):
        self.iteration = 0  # resets iteration counter
        self.SSE = np.infty  # and the SSE

        poolX = range(self.num_cols)
        poolY = range(self.num_rows)

        # generate K random pixels (Kx2 array with X,Y coordinates in each row)
        random_pixels = np.array([np.random.choice(poolX, self.k), np.random.choice(poolY, self.k)]).T

        for label in range(self.k):
            self.means[label, :3] = self.im[random_pixels[label, 1], random_pixels[label, 0], :3]
            self.means[label, 3] = random_pixels[label, 0]
            self.means[label, 4] = random_pixels[label, 1]

        return self.SSE
        # The x value of the pixel is random_pixels[label,1] which is an array

    # This function compute average values for R, G, B, X, Y channel (feature component) at pixels in each cluster
    # represented by labels in given mask "self.labels" storing indeces in range [0,K). The averages should be
    # saved in (Kx5) matrix "self.means". The return value should be the number of non-empty clusters.
    def __compute_means(self):
        labels = self.labels

        non_empty_clusters = 0

        shape = (self.num_rows, self.num_cols)  # currently the best label for each pixel.
        meshgrid = np.meshgrid(range(0, self.num_cols), range(0, self.num_rows))
        a = meshgrid[0]
        b = meshgrid[1]

        imXY = np.dstack((b, a))
        # Your code below should compute average values for R,G,B,X,Y features in each segment
        # and save them in (Kx5) matrix "self.means". For empty clusters set the corresponding mean values
        # to infinity (np.infty). Report the correct number of non-empty clusters by the return value.

        # use self.means

        # Calculate the average value of all vectors such that self.im[labels == label]
        for label in range(self.k):
            labels_eq_label = (labels == label)
            # labels_eq_label a m x n array of true and false values
            if np.all(labels_eq_label == False):
                self.means[label, :] = np.infty
                continue;

            non_empty_clusters = non_empty_clusters + 1

            # This could be a problem, Calculate the average value of all the pixels that are under labels_eq_label
            self.means[label, : 3] = np.average(self.im[labels_eq_label], axis=0)
            self.means[label, 3] = np.average(imXY[labels_eq_label, 0])
            self.means[label, 4] = np.average(imXY[labels_eq_label, 1])

        return non_empty_clusters
    # This function computes optimal (cluster) index/label in range 0,1,...,K-1 for pixel x,y based on

    # given current cluster means (self.means). The functions should save these labels in "self.labels".
    # The return value should be the corresponding optimal SSE.
    # the error between the pixels is calculated as the difference between the intensity values of the pixels in the RGB space
    # The distance between two pixels is calculated as the Euclidean distance between the two points in the RGB space
    def __compute_labels(self):
        shape = (self.num_rows, self.num_cols)
        opt_labels = np.full(shape, fill_value=self.no_label,
                             dtype=np.uint8)  # HINT: you can use this array to store and update
        # currently the best label for each pixel.
        meshgrid = np.meshgrid(range(0, self.num_cols), range(0, self.num_rows))
        a = meshgrid[0]
        b = meshgrid[1]

        imXY = np.dstack((b, a))

        min_dist = np.full(shape, fill_value=np.inf)  # HINT: you can use this array to store and update
        # the (squared) distance from each pixel to its current "opt_label".
        # use 'self.w' as a relative weight of sq. errors for X and Y components

        # Replace the code below by your code that computes "opt_labels" array of labels in range [0,K) where
        # each pixel's label is an index 'i' such that self.mean[i] is the closest to R,G,B,X,Y values of this pixel.
        # Your code should also update min_dist so that it contains the optmail squared errors

        for i in range(self.k):
            norm_im_diff = (self.im - self.means[i][0:3]) * (self.im - self.means[i][0:3])
            imXY_diff = self.w * (imXY - self.means[i][3:5]) * (imXY - self.means[i][3:5])
            # min_candidate = np.full(shape, fill_value=np.inf)
            min_candidate = np.sum(norm_im_diff, axis=2) + np.sum(imXY_diff, axis=2) * self.dist_sensitve
            min_candidate = np.sqrt(min_candidate)

            # print(np.sum(imXY_blyat, axis=2))

            opt_labels[min_candidate < min_dist] = i
            min_dist = np.minimum(min_dist, min_candidate)

        # update the labels based on opt_labels computed above
        self.labels = opt_labels
        # returns the optimal SSE (corresponding to optimal clusters/labels for given means)
        return np.sqrt(min_dist.sum())

    # The segmentation mask is used by KmeanPresenter to paint segments in distinct colors
    # NOTE: valid region labels are in [0,K), but the color map in KmeansPresenter
    #       accepts labels in range [0,K] where pixels with no_label=K are not painted/colored.
    def get_region_mask(self):
        return self.labels
    # The function below is called by "on_key_down" in KmeansPresenter".
    # It's goal is to run an iteration of K-means procedure
    # updating the means and the (segment) labels
    def iterate(self):
        self.iteration += 1

        # the main two steps of K-means algorithm
        SSE = self.__compute_labels()
        num_clusters = self.__compute_means()

        # computing improvement and printing some information
        num_pixels = self.num_rows * self.num_cols
        improve_per_pixel = (self.SSE - SSE) / num_pixels
        energy_per_pixel = SSE / num_pixels
        self.SSE = SSE
        '''
        print('iteration = {:_>2d},  clusters = {:_>2d},  SSE/p = {:_>7.1f},   improve/p = {:_>7.3f}    '.format(
            self.iteration, num_clusters, energy_per_pixel, improve_per_pixel))
        '''
        # also print the SSE per pixel print in the f" format
        print(f'iteration = {self.iteration},  clusters = {num_clusters},  SSE/p = {energy_per_pixel},   improve/p = {improve_per_pixel}    ')
        return improve_per_pixel
    def run(self, max_iter=100, min_improve_per_pixel=1e-6):
        # run iterations until max_iter is reached or SSE improvement per pixel is below min_improve_per_pixel

        for i in range(max_iter):
            improve = self.iterate()

            # give an image of the region mask
            plt.imshow(self.get_region_mask())
            # save the image in a varaible
            # plt.savefig(os.path.join(plots_folder, '{}_kmeans_iter_{:02d}.png'.format(img_name, i))
            # plt.imsave(os.path.join(plots_folder, '{}_kmeans_iter_{:02d}.png'.format(img_name, i)), app.labels)
            # plt.savefig(os.path.join(plots_folder, '{}_kmeans_iter_{:02d}.png'.format(img_name, i)))
            plt.imsave(os.path.join(self.plots_folder, '{}_kmeans_iter_{:02d}.png'.format(self.img_name, i)), self.labels)
            # print('Iteration {:02d} - energy: {:.2f}'.format(i, app.energy))
            if improve < 1e-6:
                break
        return self.labels