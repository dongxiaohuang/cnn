from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
# import perfomance as pf
# Using TensorFlow backend.

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 150 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
conv_depth_3 = 128 # ...switching to 64 after the first pooling layer
conv_depth_4 = 256 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 1024 # the FC layer will have 512 neurons

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


inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='softplus')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='softplus')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Conv [128] -> Conv [128] -> Pool (with dropout on the pooling layer)
conv_5 = Convolution2D(conv_depth_3, (kernel_size, kernel_size), padding='same', activation='softsign')(drop_2)
conv_6 = Convolution2D(conv_depth_3, (kernel_size, kernel_size), padding='same', activation='relu')(conv_5)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_6)
drop_3 = Dropout(drop_prob_1)(pool_3)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_3)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

#augment the data
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True,  # apply ZCA whitening
    rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
datagen.fit(X_train)
# checkpoint
filepath="./bestmodels/weights3.{epoch:02d}-{val_loss:.2f}.hdf5"
augmented_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [augmented_checkpoint]

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),                # Train the model using the training set...
          steps_per_epoch = len(X_train) / batch_size,#TODO :check
          epochs=num_epochs,
          callbacks = callbacks_list,
          verbose=1,
          validation_data= (X_val, Y_val)) # ...holding out 10% of the data for validation
score = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
Y_predict = model.predict(X_test, batch_size=None, verbose=1, steps=None)
y_predict = np.argmax(Y_predict, axis=1)

cm = confusion_matrix(y_test,y_predict)
print(cm)
# # rec_pre = pf.recall_precision_rates(num_classes, cm)
# # f1 = pf.fa_measure(1, num_classes, rec_pre)
# # cr = pf.all_classfi_rate(cm)
#
# print(cr)
# print(f1)

print('loss : %.2f'%score[0])
print('acc : %.2f'%score[1]*100)
# checkpoint
#
