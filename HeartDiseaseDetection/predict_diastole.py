import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import glob
import pydicom
import csv


ventricle_volumes = ['Id', 'Volume']

test_path = "testing_data"

files = []
for filename in glob.glob(test_path + "/*.dcm"):
    files.append(filename)

i = 0

image_size = 128
num_channels = 3

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
        ims = []
        ims.append(image)
        ims = np.array(ims, dtype=np.uint8)
        ims = ims.astype('float32')
        ims = np.multiply(ims, 1.0 / 255.0)
        images_all.append(ims)
        images = np.vstack((img,))

    prev_dir = str_dir

    i += 1

g = 1

volume_group = []

for j in range (0, len(images_all)):

    x_batch = images_all[j].reshape(1, image_size,image_size,num_channels)

    sess = tf.Session()

    saver = tf.train.import_meta_graph('../heartdiseasedetectionmodel_diastole.meta')

    saver.restore(sess, tf.train.latest_checkpoint('../'))

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)

    ventricle_volumes.append([dirs[j] + "_Diastole", result[0][0]])

    g += 1

csv.register_dialect('volumeDialect', delimiter=',', quoting=csv.QUOTE_NONE, escapechar="\\")

volumeFile = open('volume_diastole.csv', 'w', newline='')
with volumeFile:
   writer = csv.writer(volumeFile, dialect='volumeDialect')
   writer.writerows(ventricle_volumes)

