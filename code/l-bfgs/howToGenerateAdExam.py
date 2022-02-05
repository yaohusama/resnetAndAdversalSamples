# pre-req: cfinder.py in the same directory

#import library
import tensorflow as tf     #cpu tensorflow   (may work on gpu)
from tensorflow import keras        #use keras as frontend
import os
# Helper libraries
import numpy as np
import scipy.optimize as op
from sklearn.preprocessing import OneHotEncoder
from cfinder import cfind

import pickle
#load keras mnist dataset  (or any data set you like)
# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
training_file = '../data/train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

onehotencode=OneHotEncoder()
y_train=(y_train.reshape(-1,1))
# print(y_train)
# y_train=(y_train).toarray()
# print(y_train)
train_images=X_train
train_labels=y_train
testing_file = '../data/test.p'
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']

y_test=(y_test.reshape(-1,1))
# y_test=y_test.toarray()
test_images=X_test
test_labels=y_test
valid_file = '../data/valid.p'
with open(valid_file, mode='rb') as f:
    test = pickle.load(f)
X_validate, y_validate = test['features'], test['labels']

y_validate=(y_validate.reshape(-1,1))
# y_validate=y_validate.toarray()

#generate flattened dataset (not used in training, code optimize model to incoporate this, could be a big performance boost)
flat_train = train_images.reshape((-1,32*32*3))#28*28
flat_test = test_images.reshape((-1,32*32*3))#28*28


#normalize the dataset to range [0, 1]
train_images = train_images/255.0
test_images = test_images/255.0

#simple fully-connected neural network creation
def createModel():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),#28,28
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(43, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


#checkpoint saving callback initialization   (address format may change based on different operating system)
checkpoint_path = "training_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#here the best practice is not to train the model every time, but train the model and save the directory
#pass directory to cfind object to get model weight
#create model and fit on the traning dataset
model = createModel()
model.fit(train_images, train_labels, epochs=10, callbacks = [cp_callback])


#create cfinding object
#only using the first 10 samples to save time
cf = cfind(test_images, test_labels, createModel(), checkpoint_path)

cf.test_initialize()

cf.findAd()

#access the samples and c values, you can either save them or directly use them
samples = cf.getAdvSample()
x={"features":samples}
f=open("bfgs.pkl","wb")
pickle.dump(x,f)
print(samples.shape)
cs = cf.getC()

# print(cs)

# r = (samples-test_images[0:11]).reshape((11,784))
# v = np.matmul(r, np.transpose(r))/784
# f = np.sum(np.sqrt(v.diagonal()))/11
# print(f)