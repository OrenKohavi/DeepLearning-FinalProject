#This file defines the actual neural network
import tensorflow as tf

class BlockCNN(tf.keras.Model):
    def __init__(self):
        super(BlockCNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=3, padding='same')
        self.residual_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides = 1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            #... Repeat until residual layer is complete
        ])
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.residual_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides = 1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            #... Repeat until residual layer is complete
        ])
        self.conv3 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.residual_1(x)
        x = self.conv2(x)
        x = self.residual_2(x)
        x = self.conv3(x)
        return x
    
    def loss(self, labels, inputs):
        #Temporarily, juse use MSE loss
        #Later, this needs to be residual as defined in the paper
        return tf.keras.losses.mse(inputs, labels)