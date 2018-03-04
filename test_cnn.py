from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
#import perfomance as pf
# Using TensorFlow backend.

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 200 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.7 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)
    #TODO: remove the slides
X_train_t = data['X_train']
y_train_t = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#TODO: y_train to array
num_train, height, width, depth = X_train_t.shape # there are 28709 training examples
num_test = X_test.shape[0] #number of test exapmles
num_classes = np.unique(y_train_t).shape[0] # there are 7 image classes

X_train_t = X_train_t.astype('float64')
X_test = X_test.astype('float64')
X_train_t /= np.max(X_train_t) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train_t = np_utils.to_categorical(y_train_t, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

X_train = X_train_t[0:int(num_train*.9)]
Y_train = Y_train_t[0:int(num_train*.9)]
X_val = X_train_t[int(num_train*.9+1) :]
Y_val = Y_train_t[int(num_train*.9+1) :]


###########################################load model#####################
model_resume = load_model('./bestmodels/weights.119-1.23.hdf5')

score = model_resume.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
Y_predict = model_resume.predict(X_test, batch_size=None, verbose=1, steps=None)
y_predict = np.argmax(Y_predict, axis=1)

cm = confusion_matrix(y_test,y_predict)
print(cm)
#rec_pre = pf.recall_precision_rates(num_classes, cm)
#f1 = pf.fa_measure(1, num_classes, rec_pre)
#cr = pf.all_classfi_rate(cm)

#print(cr)
#print(f1)

print('loss : %.2f'%score[0])
print('acc : %.2f'%score[1]*100)
# checkpoint
#
