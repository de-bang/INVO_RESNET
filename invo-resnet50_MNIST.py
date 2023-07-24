#IMPORTING LIBRARIES
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import math
import datetime
import platform
import cv2
print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline
import tensorflow.keras.backend as K
# Image dataset has channels as its last dimensions
K.set_image_data_format('channels_last')


#DATASET (MNIST)
mnist_dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
# Save image parameters to the constants that we will use later for data re-shaping and for model traning.
(Number_of_elements, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
IMAGE_CHANNELS = 1
x_train = np.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [32,32])
x_test = np.expand_dims(x_test, axis=-1)
x_test = tf.image.resize(x_test, [32,32])
x_train_normalized = x_train / 255
x_test_normalized = x_test / 255
# print(x_train_normalized.shape)
# print(x_test_normalized.shape)



#MODEL CREATION

##INVOLUTION BLOCK
import tensorflow as tf
class Involution(keras.layers.Layer):
    def __init__(
        self, channel, group_number, kernel_size, stride, reduction_ratio, name=None):
        super().__init__(name=name)

        # Initialize the parameters.
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def get_config(self):
     #   config = super().get_config()
        config=({
                  'channel': self.channel,
                  'group_number': self.group_number,
                  'reduction_ratio': self.reduction_ratio,
                  'kernel_size': self.kernel_size,
                  'stride': self.stride
                  })
        return config


    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output

##IDENTITY BLOCK
def identity_block(X, f, filters, stage, block):

    inv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X
    #X=Involution(channel=F1,group_number=1,kernel_size=1,stride=1,reduction_ratio=1, name="inv_01_"+bn_name_base + '2a')(X)
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name="conv1" + bn_name_base+ '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X=Involution(channel=F2,group_number=1,kernel_size=f,stride=1,reduction_ratio=1, name="inv_02_"+bn_name_base + '2b')(X)
    #X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=inv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #X=Involution(channel=F3,group_number=1,kernel_size=1,stride=1,reduction_ratio=1, name="inv_03_"+bn_name_base + '2c')(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name="conv3" + bn_name_base +'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X

##CONVOLUTION BLOCK
def convolutional_block(X, f, filters, stage, block, s=2):

    inv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    #X_shortcut = X
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='same', name="conv1" + bn_name_base+ '2a'+"shortcut", kernel_initializer=glorot_uniform(seed=0))(X)
    #X=Involution(channel=F1,group_number=1,kernel_size=1,stride=s,reduction_ratio=1, name="inv_1_"+bn_name_base + '2a')(X)
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name="conv1" + bn_name_base+ '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X=Involution(channel=F2,group_number=1,kernel_size=f,stride=1,reduction_ratio=1, name="inv_2_"+bn_name_base + '2b')(X)
    print(1)
    print(X.shape, X_shortcut.shape)
    #X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #X=Involution(channel=F3,group_number=1,kernel_size=1,stride=1,reduction_ratio=1, name="inv_3_"+bn_name_base + '2c')(X)
    print(2)
    print(X.shape, X_shortcut.shape)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name="conv3" + bn_name_base +'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut=Involution(channel=F3,group_number=1,kernel_size=1,stride=s,reduction_ratio=1,  name="inv_4_"+bn_name_base + '2d')(X_shortcut)

    #X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    print(3)
    print(X.shape, X_shortcut.shape)
    X = Add()([X, X_shortcut])
    print(4)
    X = Activation('relu')(X)

    return X

##RESNET50 BLOCK
def ResNet50(input_shape=(32, 32, 1), classes = 10, n=16):

    X_input = Input(input_shape)

    #X = ZeroPadding2D((3, 3))(X_input)
    #X = Conv2D(64, (7, 7), strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X=Involution(channel=64//n,group_number=1,kernel_size=7,stride=2,reduction_ratio=1, name="inv_0")(X_input)

    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64//n, 64//n, 256//n], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64//n, 64//n, 256//n], stage=2, block='b')
    X = identity_block(X, 3, [64//n, 64//n, 256//n], stage=2, block='c')



    X = convolutional_block(X, f=3, filters=[128//n, 128//n, 512//n], stage=3, block='a', s=2)

    X = identity_block(X, 3, [128//n, 128//n, 512//n], stage=3, block='b')
    X = identity_block(X, 3, [128//n, 128//n, 512//n], stage=3, block='c')
    X = identity_block(X, 3, [128//n, 128//n, 512//n], stage=3, block='d')


    X = convolutional_block(X, f=3, filters=[256//n, 256//n, 1024//n], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256//n, 256//n, 1024//n], stage=4, block='b')
    X = identity_block(X, 3, [256//n, 256//n, 1024//n], stage=4, block='c')
    X = identity_block(X, 3, [256//n, 256//n, 1024//n], stage=4, block='d')
    X = identity_block(X, 3, [256//n, 256//n, 1024//n], stage=4, block='e')
    X = identity_block(X, 3, [256//n, 256//n, 1024//n], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512//n, 512//n, 2048//n], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512//n, 512//n, 2048//n], stage=5, block='b')
    X = identity_block(X, 3, [512//n, 512//n, 2048//n], stage=5, block='c')

    X = AveragePooling2D(pool_size=(1, 1), padding='same')(X)



    # output layer
    X = Flatten()(X)
    X = Dense(256, activation='relu', name='fc_d1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    return model

model = ResNet50()
model.summary()

tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
)


#MODEL TRAINING
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(
    optimizer=adam_optimizer,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

trining_history = model.fit(x_train_normalized,
     y_train,
     epochs=100,
     batch_size = 256,
     validation_data=(x_test_normalized, y_test),
 )


#MODEL EVALUATION
train_loss, train_accuracy = model.evaluate(x_train_normalized, y_train)
print('Training loss: ', train_loss)
print('Training accuracy: ', train_accuracy)
validation_loss, validation_accuracy = model.evaluate(x_test_normalized, y_test)
print('Validation loss: ', validation_loss)
print('Validation accuracy: ', validation_accuracy)
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()










