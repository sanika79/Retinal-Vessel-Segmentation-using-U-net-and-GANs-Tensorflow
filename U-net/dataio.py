import os
from os import path

import cv2 as cv
from cv2 import imread

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


class DataIO:

    def load_matfile_images_first(self, file):
        """
        Takes in the path to a file and returns a numpy array of images.

        :param file: relative path to data file. Expected format is .mat.
        :return:  np.nparray of (data, labels) where each data and labels is in the following format:
            (number of samples, image height, image width, image channels)
        """

        assert '.mat' in file, 'Provide a valid .mat file'

        # should populate the following fields #todo
        # self.num_data = 1040  # number of images
        self.filepath = file  # dont need to change this

        image_train = h5py.File(file, 'r')  # Reads the transpose of order of columns

        image = image_train['image']
        image = np.array(image)
        # image = tf.cast(image, tf.float32)

        label_train = h5py.File(file)  # Reads the transpose of order of columns
        label = label_train['label']
        label = np.array(label)
        # label = tf.cast(label, tf.float32)

        return image, label

    def save_image_array(self, img_array: np.array, dest_file: str, img_key: str = 'images'):
        """
        Saves the array of images as a matfile to the given destination file.
        Args:
            img_key: in the resulting matfile, what to call the given array.
            img_array: input np.array of images to save
            dest_file: destination filepath to write to

        Returns:
            None, saves to disk
        """
        assert not path.exists(dest_file), 'Destination file already exists.'

        savemat(dest_file, {img_key: img_array})

    def save_image_folder(self, img_array: np.array, dest_folder: str):
        """
        todo fill this in
        Saves the array of images as a matfile to the given destination file.
        Args:
            img_array: input np.array of images to save
            dest_folder: destination filepath to write to

        Returns:
            None, saves to disk
        """
        pass

class ImageHandler:
    def __init__(self):
        pass

    def display_labels_masks_first(self, image_list, title='Image'):
        image_list = np.array([x.squeeze() for x in image_list])
        for e in image_list:
            self.display_image(e, title)

    def display_labels_masks_last(self, image_list, title='Image'):
        """
        Horizontally stacks and displays multiple images
        :param image_list:  list of images
        :param title: title to use for the single output image
        :return: none, displays images
        """
        image_list = np.array([x.squeeze() for x in image_list])
        # self.disp(np.concatenate(imglist, axis=1), title)

        for x in range(np.size(image_list, 2)):
            self.display_image(image_list[:, :, x], title)

    def display_image(self, img, title='Image'):
        """
        :param img: image to show.
        :return: None, should render to cell.

        Args:
            title:
        """

        # change from (x, y, 1) --> (x, Y)
        img = img.squeeze()

        # show image
        plt.imshow(img, cmap='gray')
        # set title
        plt.title(title)

        plt.imsave(title+'.jpg',img,cmap='gray')

        plt.show()



        # allows showing multiple images in one cell for Jupyter
        # plt.figure()

class ImageHelper:
    @staticmethod
    def read_folder(file_path: str, pattern='.png'):
        """
        Reads in the contents of a folder defined by path and returns a list of all files that match the given file extension pattern.
        Args:
            file_path: path to folder to read.
            pattern: extension of the file. Must start with '.'

        Returns:
            List of all files in directory that match given pattern.
        """
        assert pattern[0] == '.', 'Enter valid extension'
        assert path.exists(file_path), 'Enter a valid directory'
        files = os.listdir(file_path)

        return [file_path + os.sep + x for x in files if pattern in x]

    @staticmethod
    def convert_images_to_np(file_path):
        """
        Converts a list of images
        Args:
            file_path: folder containing all images

        Returns:
            Array of images first, each image is (height, width, num_channels).
        """

        image_list = ImageHelper.read_folder(file_path, '.png')
        images = []

        for i in image_list:
            images.append(imread(i, cv.IMREAD_GRAYSCALE))

        numpy_arr = np.array(images)

        return numpy_arr

    @staticmethod
    def convert_and_save_spine_dataset():
        """
        Function to convert and save the folder of spine images into a training matfile of images and labels.

        Returns: None, writes to disk.
        """
        # configure these
        training_data = 'spine_dataset\\Images'
        test_data = 'spine_dataset\\Labels1'
        destination = 'spine_dataset_training_source.mat'

        images = ImageHelper.convert_images_to_np(training_data)
        images.shape = tuple(list(images.shape) + [1])
        labels = ImageHelper.convert_images_to_np(test_data)
        labels.shape = tuple(list(images.shape) + [1])

        ImageHelper.save_h5py(destination, {'image': images, 'label': labels})

    @staticmethod
    def save_h5py(name: str, data: dict):
        """
        Saves a dictionary of data to a h5 file.
        Args:
            name: name of the output file.
            data: dict of data to give to the file.

        Returns:

        """
        file = h5py.File(name, 'w')
        for k, v in data.items():
            file.create_dataset(k, data=v)

        file.close()


