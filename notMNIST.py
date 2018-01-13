# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:20:57 2017

@author: austi
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils


#from keras import backend as K
#K.set_image_dim_ordering('th')





url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


# Code from Udacity to extract the data files
num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


#Prepare Dataset to be more maangales using code from Udacity Deep Leraning project
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      print('this might take a while')
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 50000)
test_datasets = maybe_pickle(test_folders, 2500)



##take the data, and make a training/validation/testing set out of it.  Code from Udacity Deep Leearning course

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 400000
valid_size = 15000
test_size = 15000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


## Randomzies the order of the dataset.  Code from Udacity Deep Learing Course
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
'''
#convert dataset from (nb,w,h) to (nb,w,h,c)
train_dataset_kr = np.reshape(train_dataset, (train_dataset.shape[0], 28, 28,1))
test_dataset_kr = np.reshape(test_dataset, (test_dataset.shape[0],28,28,1))
valid_dataset_kr = np.reshape(valid_dataset,(valid_dataset.shape[0],28,28,1))

#converts labels to catagorial 
train_labels_kr = np_utils.to_categorical(train_labels, 10)
test_labels_kr = np_utils.to_categorical(test_labels,10)
vaild_labels_kr = np_utils.to_categorical(valid_labels,10)

#convert dataframe to tensor
#train_dataset_tf = tf.convert_to_tensor(train_dataset,np.float32)
#add color depth to image data

#train_dataset_tf = train_dataset_tf.reshape(train_dataset_tf.shape[0], 1, 28, 28))

#print out content of tensor
#sess = tf.InteractiveSession() 
#print(train_dataset_tf.eval())



#Modeling of job in kreas 

model = Sequential()
 
model.add(Convolution2D(8, 3, strides = 3, activation='relu', input_shape=(28,28,1),data_format = 'channels_last',padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(16, 3, strides = 3, activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, strides = 3, activation='relu',padding='same'))
#model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Training the model in kreas
history = model.fit(train_dataset_kr,train_labels_kr,epochs = 500, validation_data = (valid_dataset_kr,vaild_labels_kr),batch_size = 256)
#model.train_on_batch(train_dataset_kr,train_labels_kr)

score = model.evaluate(test_dataset_kr, test_labels_kr, verbose=0)
#test_per = model.predict(test_dataset_kr)
'''


def cnn_model_fn(features,labels,mode):
    #Input Layer
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    
    #Covolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=8,
            kernel_size=[3,3],
            strides = (3,3),
            padding = 'same',
            activation = tf.nn.relu)
    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1, 
            pool_size=[2,2], 
            strides = 2,
            padding = 'same')
    
    #Convlolutional Layer #2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters=16,
            kernel_size = [3,3],
            strides = 3,
            padding = 'same')
    
    #Pooling Lyaer #2
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2,
            pool_size=[2,2],
            strides = 2,
            padding = 'same')
    
    
   
    #Flatten the input
    flat = tf.reshape(
            pool2,
            [-1, 7 * 7 * 16])
    
    #Create a Dense Layer with 256 Neurons 
    dense392 = tf.layers.dense(
            inputs = flat,
            units=392,
            activation = tf.nn.relu)
    
    #Does Dropout at 50%
    dropout_dense392 = tf.layers.dropout(
            inputs = dense392,
            rate = 0.5,
            training=mode == tf.estimator.ModeKeys.TRAIN)
    
    #Creates a Dense Layer with 128 Neuros
    dense196 = tf.layers.dense(
            inputs = dropout_dense392,
            units = 196,
            activation = tf.nn.relu)
    
    #Does Droupout at 50%
    dropout_dense196 = tf.layers.dropout(
            inputs = dense196,
            rate = 0.5,
            training=mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(
            inputs = dropout_dense196,
            units = 10)
    
    
    
    predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDecentOptimizer(learning_rate=0.001)
        train_op= optimizer.minimize(
                loss = loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode = mode,
                loss = loss,
                train_op = train_op)
        
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                        labels=labels,
                        predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops = eval_metric_ops)
    
    
    
    
not_mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/model")    

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)


#training the Tensorflow model
train_inputs = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_dataset },
        y=train_labels,
        batch_size=98,
        num_epochs=None,
        shuffle=True)
not_mnist_classifier.train(
        input_fn=train_inputs,
        steps=100,
        hooks=[logging_hook])

#Evaluate

test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_dataset},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
test_results = not_mnist_classifier.evaluate(input_fn=test_input_fn)
print(test_results)

