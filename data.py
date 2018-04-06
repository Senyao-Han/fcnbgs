from __future__ import print_function
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
import time

class DataLoader:
    def __init__(self, data_dir, batch_size, image_shape):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.width, self.height = image_shape

        # get subdirs
        self.series = [o for o in os.listdir(self.data_dir) if os.path.isdir(self.data_dir+o)]
        self.series = sorted(self.series)

        # get lengths
        self.num_series = len(self.series)
        self.lengths = [len(os.listdir(self.data_dir+x+'/groundtruth')) for x in self.series]
        self.seq = [np.arange(1, l+1) for l in self.lengths]
        for seq in self.seq: np.random.shuffle(seq)
        self.num_images = 0
        for l in self.lengths: self.num_images += l
        self.num_batches = self.num_images / self.batch_size

        #initialize indices
        self.series_idx = 0
        self.idx = [0 for x in self.series]
        self.batch_idx = 0

        #threads
        self.num_threads = 8
        self.each_thread = self.batch_size/self.num_threads

    def reset(self):
        self.series_idx = 0
        self.idx = [0 for x in self.series]
        self.batch_idx = 0

    def next_batch_threading(self):
        batch_x = np.zeros([self.batch_size, self.height, self.width, 3], dtype='uint8')
        batch_y = np.zeros([self.batch_size, self.height, self.width, 1], dtype='uint8')

        threads = [self.LoadImageThread(self, self.each_thread, batch_x, batch_y) for i in range(self.num_threads)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()

        self.batch_idx += 1
        return batch_x, batch_y

    def next_batch(self):
        batch_x = np.zeros([self.batch_size, self.height, self.width, 3], dtype='uint8')
        batch_y = np.zeros([self.batch_size, self.height, self.width, 1], dtype='uint8')

        for i in range(self.batch_size):
            Ix, Iy = self.next_image()
            batch_x[i, :, :, :] = Ix
            batch_y[i, :, :, :] = Iy

        self.batch_idx += 1
        return batch_x, batch_y

    class LoadImageThread(threading.Thread):
        def __init__(self, parent, n, batch_x, batch_y):
            threading.Thread.__init__(self)
            self.parent = parent
            self.n = n
            self.batch_x = batch_x
            self.batch_y = batch_y
        def run(self):
            for i in range(self.n):
                self.batch_x[i, :, :, :], self.batch_y[i, :, :, :] = self.parent.next_image()

    def next_image(self):
        series = self.series[self.series_idx]
        idx = int(self.seq[self.series_idx][self.idx[self.series_idx]])
        x_dir = self.data_dir + series + '/' + 'input/'
        y_dir = self.data_dir + series + '/' + 'groundtruth/'
        x_path = x_dir + 'in%06d.jpg'%idx
        y_path = y_dir + 'gt%06d.png'%idx

        try:
            Ix = cv2.imread(x_path)
            Ix = cv2.resize(Ix, (self.width, self.height))

            Iy_ = cv2.imread(y_path)
            Iy_ = cv2.resize(Iy_, (self.width, self.height))

            Iy_ = cv2.cvtColor(Iy_, cv2.COLOR_BGR2GRAY)
            Iy = np.zeros_like(Iy_, dtype='float32')
            Iy[Iy_ == 255] = 1
            Iy = np.expand_dims(Iy, axis=2)
        except:
            print(x_path)
            print(y_path)

        #update indices
        if self.idx[self.series_idx] < self.lengths[self.series_idx]-1:
            self.idx[self.series_idx] += 1
        for i in range(1, len(self.series)-1):
            series_idx = (self.series_idx + i) % self.num_series
            if self.idx[series_idx] < self.lengths[series_idx]-1:
                self.series_idx = series_idx
                break

        return Ix, Iy