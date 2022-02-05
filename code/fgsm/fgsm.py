import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
batch_size=256


def LeNet(x, KEEP_PROB):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Input = 32x32x3. Output = 28x28x6.
    # Convolutional.
    conv1_w = tf.Variable(tf.truncated_normal((5, 5, 3, 6), mu, sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, [1, 1, 1, 1], 'VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    # Layer 2: Input = 14x14x6. Output = 10x10x16.
    # Convolutional.
    conv2_w = tf.Variable(tf.truncated_normal((5, 5, 6, 16), mu, sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool1, conv2_w, [1, 1, 1, 1], 'VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(pool2)

    # Layer 3: Input = 400. Output = 120.
    # Fully Connected.
    full1_w = tf.Variable(tf.truncated_normal((400, 120), mu, sigma))
    full1_b = tf.Variable(tf.zeros(120))
    full1 = tf.matmul(flat, full1_w) + full1_b
    # Activation.
    full1 = tf.nn.relu(full1)
    # Dropout
    full1 = tf.nn.dropout(full1, KEEP_PROB)
    # Layer 4: Input = 120. Output = 84.
    # Fully Connected.
    full2_w = tf.Variable(tf.truncated_normal((120, 84), mu, sigma))
    full2_b = tf.Variable(tf.zeros(84))
    full2 = tf.matmul(full1, full2_w) + full2_b
    # Activation.
    full2 = tf.nn.relu(full2)
    # Dropout
    full2 = tf.nn.dropout(full2, KEEP_PROB)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    full3_w = tf.Variable(tf.truncated_normal((84, 43), mu, sigma))
    full3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(full2, full3_w) + full3_b

    return logits
#STEP 2 - Architecture selection
# Here all the DNN architecture is created
# define a simple CNN network
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
#
# model = Sequential()
#
# # add Con2D layers
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)))
# model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
#
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
#
# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
#
# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
#
# # flatten
# model.add(Flatten())
#
# # dropOut layer
# model.add(Dropout(0.2))
#
# # add one simple layer for classification
# model.add(Dense(units=512, activation='relu'))
#
# # add output layer
# model.add(Dense(units=43, activation='softmax'))
#
# # compile model
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
#
# # print model info
# model.summary()
# json_str = model.to_json()
# print(json_str)
def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)#,padding='valid
    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)#,padding="valid"
    # with tf.variable_scope('conv2'):
    #     z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    #     z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=1, padding="valid")
    # with tf.variable_scope('conv3'):
    #     z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    #     z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=1,padding="valid")
    with tf.variable_scope('flat'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])
    # if training:
    #     z=tf.layers.dropout(z,rate=0.2)
    # z = tf.layers.dense(z, units=512, name='logits0',activation="relu")
    l_layer = tf.layers.dense(z, units=43, name='logits')
    y = tf.nn.softmax(l_layer, name='ybar')
    if logits:
        return y, l_layer
    return y

#FGSM
def fgm(model, x, eps=0.01, epochs=1, sign=True, clip_min=0, clip_max=1):
    xadv = tf.identity(x)
    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]
    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
     tf.equal(ydim,1),
     lambda: tf.nn.relu(tf.sign(ybar-0.5)),
     lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))
    loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2
    noise_fn = tf.sign
    eps = tf.abs(eps)
    # eps=0
    def cond(xadv, i):
        return tf.less(i, epochs)
    def body(xadv, i):
        ybar,logits= model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        # xadv=tf.stop_gradient(xadv)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1
    xadv, _ = tf.while_loop(cond, body, (xadv, 0), back_prop=False, name='fast_gradient')
    return xadv

# CLASS ENVIRONMENT DEFINITION, BEFORE RUNNING MAIN
class Environment():
	pass

ambiente = Environment()

with tf.variable_scope('model'):
    ambiente.x = tf.placeholder(tf.float32, (None, 32,32, 3))
    ambiente.y = tf.placeholder(tf.float32, (None, 43), name='y')
# calls model (STEP 2)
    ambiente.ybar,logits = model(ambiente.x,logits=True)
    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(ambiente.y, axis=1), tf.argmax(ambiente.ybar, axis=1))
        ambiente.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ambiente.y, logits=logits)
        ambiente.loss = tf.reduce_mean(cross_entropy, name='loss')
    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        ambiente.train_op = optimizer.minimize(ambiente.loss)
with tf.variable_scope('model', reuse=True):
    ambiente.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    ambiente.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    ambiente.x_fgsm = fgm(model, ambiente.x, epochs=ambiente.fgsm_epochs, eps=ambiente.fgsm_eps)

#STEP 4 - Training
def training(sess, ambiente, X_data, Y_data, X_valid=None, y_valid=None, shuffle=True, batch=128, epochs=1):
    Xshape = X_data.shape
    n_data = Xshape[0]
    n_batches = int(n_data/batch)
    # print(n_batches)
    # print(X_data.shape)
    for ep in range(epochs):
        print('epoch number: ', ep+1)
        if shuffle:
            ind = np.arange(n_data)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            Y_data = Y_data[ind]
        for i in range(n_batches):
            # print("y")
            print(' batch {0}/{1}'.format(i + 1, n_batches),end="\r")
            start = i*batch
            end = min(start+batch, n_data)
            sess.run([ambiente.train_op], feed_dict={ambiente.x: X_data[start:end], ambiente.y: Y_data[start:end]})
            # print(ambiente.loss)
        evaluate(sess, ambiente, X_data, Y_data)
		
def evaluate(sess, ambiente, X_test, Y_test, batch=128):
	n_data = X_test.shape[0]
	n_batches = int(n_data/batch)
	totalAcc = 0
	totalLoss = 0
	for i in range(n_batches):
		print(' batch {0}/{1}'.format(i + 1, n_batches), end='\r')
		start = i*batch 
		end = min(start+batch, n_data)
		batch_X = X_test[start:end]
		batch_Y = Y_test[start:end]
		batch_loss, batch_acc = sess.run([ambiente.loss, ambiente.acc], feed_dict={ambiente.x: batch_X, ambiente.y: batch_Y})
		totalAcc = totalAcc + batch_acc*(end-start)
		totalLoss = totalLoss + batch_loss*(end-start)
	totalAcc = totalAcc/n_data
	totalLoss = totalLoss/n_data
	print('acc: {0:.3f} loss: {1:.3f}'.format(totalAcc, totalLoss))
	return totalAcc, totalLoss


def perform_fgsm(sess, ambiente, X_data, epochs=1, eps=0.01, batch_size=128):
    print('\nInizio FGSM')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(ambiente.x_fgsm, feed_dict={
            ambiente.x: X_data[start:end],
            ambiente.fgsm_eps: eps,
            ambiente.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()
    return X_adv


def main():
#STEP 1 - Initial Dataset Collection 
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
# read images from dataset
#     mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
#     X_train = mnist.train.images
#     y_train = mnist.train.labels
#     X_test = mnist.test.images
#     y_test = mnist.test.labels
    training_file = './data/gtsrb/train.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']

    onehotencode=OneHotEncoder()
    y_train=onehotencode.fit_transform(y_train.reshape(-1,1))
    # print(y_train)
    y_train=(y_train).toarray()
    # print(y_train)
    testing_file = './data/gtsrb/test.p'
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    X_test, y_test = test['features'], test['labels']

    y_test=onehotencode.fit_transform(y_test.reshape(-1,1))
    y_test=y_test.toarray()
    valid_file = './data/gtsrb/valid.p'
    with open(valid_file, mode='rb') as f:
        test = pickle.load(f)
    X_validate, y_validate = test['features'], test['labels']

    y_validate=onehotencode.fit_transform(y_validate.reshape(-1,1))
    y_validate=y_validate.toarray()
    tf.logging.set_verbosity(old_v)
# 90% of dataset is training set, 10% is validation set
#     i = int(X_train.shape[0] * 0.9)
#     X_validate = X_train[i:]
#     X_train = X_train[:i]
#     y_validate = y_train[i:]
#     y_train = y_train[:i]
# start tensorflow session
# runs STEP 2

    sess = tf.InteractiveSession() # ENVIRONMENT -> MODEL -> FGM

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver=tf.train.Saver(max_to_keep=1)
# runs training and evaluating
# STEP 4
    if os.path.exists(r"F:\downloads\DCGAN-tensorflow-master\BlackBoxAttackDNN-master_fgsm_jsma_tf1\ckpt"):
        print("loaded")
        saver.restore(sess, "ckpt/fgsm.ckpt")
    # training(sess, ambiente, X_train, y_train, X_validate, y_validate, shuffle=False, batch=batch_size, epochs=60)


    saver.save(sess,'ckpt/fgsm.ckpt')
    evaluate(sess, ambiente, X_test, y_test)

    X_adv = perform_fgsm(sess, ambiente, X_test/255.0, eps=0.02, epochs=12)
    X_adv=X_adv*255.0
    X_adv=X_adv.astype(int)
    x={"features":X_adv}
    filename_what=open("fgsm.pkl","wb")
    pickle.dump(x,filename_what)

    evaluate(sess, ambiente, X_adv, y_test)

if __name__ == "__main__":
    main()

# MAIN:
# STEP 1
# DATASET COLLECTION
# STEP 2
# INTERACTIVE SESSION -> ENVIRONMENT:
# MODEL
# FGM (ADVERSARIAL MODEL)
# STEP 3
# LABELING 
# STEP 4
# TRAINING 
# EVALUATE
# PERFORM_FGSM
# EVALUATE
# STEP 5
# AUGMENTATION
