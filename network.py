#This file defines the actual neural network
import tensorflow as tf

class BottleNeck(tf.keras.Model):
    def __init__(self, planes):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=1, activation=tf.keras.layers.LeakyReLU(), use_bias=False)
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(), use_bias=False)
        self.conv3 = tf.keras.layers.Conv2D(planes, kernel_size=1, activation=tf.keras.layers.LeakyReLU(), use_bias=False)
    
    def call(self, inputs):
        residual = inputs

        out = self.conv1(inputs)

        out = self.conv2(out)

        out = self.conv3(out)

        out += residual
        out = tf.nn.leaky_relu(out, 0.1)
        return out
		
class BlockCNN(tf.keras.Model):
    def __init__(self):
        super(BlockCNN, self).__init__()
        k = 16
        self.conv_1 = tf.keras.layers.Conv2D(filters=k, kernel_size=3, strides=(1,1), use_bias=False)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()
        self.layer_1 = BottleNeck(k)

        self.conv_2 = tf.keras.layers.Conv2D(filters=k*2, kernel_size=(5,5), strides=(1,1), use_bias=False)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()
        self.layer_2 = BottleNeck(k*8)

        self.conv_3 = tf.keras.layers.Conv2D(k*8, (5,5), (1,1), use_bias=False)
        self.batchnorm_3 = tf.keras.layers.BatchNormalization()
		
        self.conv_4 = tf.keras.layers.Conv2D(k, 7, 1, padding = 'valid', use_bias=False)

        self.conv_5 = tf.keras.layers.Conv2D(3, 1, 1, padding = 'same', use_bias=False)

    def call(self, inputs):
        
        out = tf.nn.leaky_relu(self.conv_1(inputs), 0.1)
        out = self.layer_1(out)
        out = tf.nn.leaky_relu(self.batchnorm_1(self.conv_2(out)), 0.1)
        out = tf.nn.leaky_relu(self.batchnorm_2(self.conv_3(out)), 0.1)
        out = self.layer_2(out)
        out = tf.nn.leaky_relu(self.batchnorm_3(self.conv_4(out)), 0.1)
        out = self.conv_5(out)
        out = tf.keras.activations.sigmoid(out)
        return out
    
    def loss(self, labels, inputs):
        return tf.keras.losses.mse(inputs, labels)