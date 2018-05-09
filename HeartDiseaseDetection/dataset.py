import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import csv
import pydicom
import scipy.io as sio


def load_train(train_path, image_size):
    images_final = []
    labels = []
    img_names = []
    cls = []

    volume_systole = []
    volume_diastole = []

    with open('training_data/train.csv', 'rt') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            volume_systole.append(row['Systole'])
            volume_diastole.append(row['Diastole'])

    files = []
    for filename in glob.glob(train_path + "/*.dcm"):
            files.append(filename)

    print(len(files))

    print('Going to read training images')

    i = 0

    images_all = []
    dirs = []
    for fl in files:

        im = pydicom.dcmread(fl)
        basefile = os.path.basename(fl)
        splitbasefile = basefile.split("_")

        str_dir = splitbasefile[1]
        img = im.pixel_array

        if (i == 0):
            prev_dir = str_dir
            images = np.vstack((img,))
            i += 1
            continue

        if (str_dir == prev_dir):
            images = np.vstack((images,img))
        else:
            dirs.append(prev_dir)
            image = np.stack((images,) * 3, -1)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images_all.append(image)
            images = np.vstack((img,))

        prev_dir = str_dir

        i += 1

    for j in range(0,len(images_all)):

        images_final.append(images_all[j])

        vol_systole = volume_systole[(int(dirs[j]) - 1)]

        label = vol_systole
        labels.append(label)
        #cls.append(vol_i-1)

    images_final = np.array(images_final)
    labels = np.array(labels)
    #cls = np.array(cls)

    return images_final, labels


class DataSet(object):

  def __init__(self, images, labels): 
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):

    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:

      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end] 


def read_train_sets(train_path, image_size, validation_size): 
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels = load_train(train_path, image_size) 
      
  images, labels = shuffle(images, labels) 
      

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]

  data_sets.train = DataSet(train_images, train_labels) 
  data_sets.valid = DataSet(validation_images, validation_labels) 

  return data_sets


