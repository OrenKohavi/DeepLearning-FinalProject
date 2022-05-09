#This file defines the actual neural network
import tensorflow as tf

class BottleNeck(tf.keras.Model):
    def __init__(self, planes, stride=1):
        super(BottleNeck, self).__init__()
        '''
        FULL NETWORK:
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=1, activation=tf.keras.layers.LeakyReLU(), use_bias=False)
        # what batchnorm? momentum = 1?
        self.bn1 = tf.keras.layers.BatchNormalization()
        # padding was 1
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, activation=tf.keras.layers.LeakyReLU(), padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(planes, kernel_size=1, activation=tf.keras.layers.LeakyReLU(), use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.stride = stride
        '''
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=1, activation=tf.keras.layers.LeakyReLU(), use_bias=False)
        # what batchnorm? momentum = 1?
		#self.bn1 = tf.keras.layers.BatchNormalization()
        # padding was 1
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(), use_bias=False)
		#self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(planes, kernel_size=1, activation=tf.keras.layers.LeakyReLU(), use_bias=False)
		#self.bn3 = tf.keras.layers.BatchNormalization()
        self.stride = stride
    
    def call(self, inputs):
        '''
        FULL NETWORK:
        residual = inputs

        out = self.conv1(inputs)   
        out = self.bn1(out)
        # slope default at .2

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.conv3(out)   
        out = self.bn3(out)

        out += residual
        out = tf.nn.leaky_relu(out, 0.1)
        '''
        residual = inputs

        out = self.conv1(inputs)
		#out = self.bn1(out)
        # slope default at .2

        out = self.conv2(out)
		#out = self.bn2(out)

        out = self.conv3(out)
		#out = self.bn3(out)

        out += residual
        out = tf.nn.leaky_relu(out, 0.1)
        return out
		
class BlockCNN(tf.keras.Model):
    def __init__(self):
        super(BlockCNN, self).__init__()
        k = 16
        #main already multiplies by 255!
        '''
        FULL NETWORK:
        k = 64
        self.conv_1 = tf.keras.layers.Conv2D(filters=k, kernel_size=3, strides=(1,1), use_bias=False)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()
        self.layer_1 = BottleNeck(k)
        self.layer_2 = BottleNeck(k)

        self.conv_2 = tf.keras.layers.Conv2D(filters=k*2, kernel_size=(5,5), strides=(1,1), use_bias=False)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()
        self.layer_3 = BottleNeck(k*2)

        self.conv_3 = tf.keras.layers.Conv2D(k*4, (5,5), (1,1), use_bias=False)
        self.batchnorm_3 = tf.keras.layers.BatchNormalization()
        
        self.layer_4 = BottleNeck(k*4)
        self.layer_5 = BottleNeck(k*4)

        self.conv_4 = tf.keras.layers.Conv2D(k*8, (5,5), (1,1), use_bias=False)
        self.batchnorm_4 = tf.keras.layers.BatchNormalization()
		
        self.layer_6 = BottleNeck(k*8)

        self.conv_5 = tf.keras.layers.Conv2D(k*4, (2,2), 1, padding = 'valid', use_bias=False)
        self.batchnorm_5 = tf.keras.layers.BatchNormalization()

        self.layer_7 = BottleNeck(k*4)

        self.conv_6 = tf.keras.layers.Conv2D(k*2, (2,2), 1, padding = 'valid', use_bias=False)
        self.batchnorm_6 = tf.keras.layers.BatchNormalization()

        self.layer_8 = BottleNeck(k*2)

        self.conv_7 = tf.keras.layers.Conv2D(k, 1, 1, padding = 'valid', use_bias=False)
        self.batchnorm_7 = tf.keras.layers.BatchNormalization()

        self.layer_9 = BottleNeck(k)

        self.conv_8 = tf.keras.layers.Conv2D(3, 1, 1, padding = 'same', use_bias=False)
        '''

        self.conv_1 = tf.keras.layers.Conv2D(filters=k, kernel_size=3, strides=(1,1), use_bias=False)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()
        self.layer_1 = BottleNeck(k)
        self.layer_2 = BottleNeck(k)

        self.conv_2 = tf.keras.layers.Conv2D(filters=k*2, kernel_size=(5,5), strides=(1,1), use_bias=False)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()
        self.layer_3 = BottleNeck(k*2)

        self.conv_3 = tf.keras.layers.Conv2D(k*4, (5,5), (1,1), use_bias=False)
        self.batchnorm_3 = tf.keras.layers.BatchNormalization()
        
        self.layer_4 = BottleNeck(k*4)
        self.layer_5 = BottleNeck(k*4)

        self.conv_4 = tf.keras.layers.Conv2D(k*8, (5,5), (1,1), use_bias=False)
        self.batchnorm_4 = tf.keras.layers.BatchNormalization()
		
        self.layer_6 = BottleNeck(k*8)

        self.conv_5 = tf.keras.layers.Conv2D(k*4, (2,2), 1, padding = 'valid', use_bias=False)
        self.batchnorm_5 = tf.keras.layers.BatchNormalization()

        self.layer_7 = BottleNeck(k*4)

        self.conv_6 = tf.keras.layers.Conv2D(k*2, (2,2), 1, padding = 'valid', use_bias=False)
        self.batchnorm_6 = tf.keras.layers.BatchNormalization()

        self.layer_8 = BottleNeck(k*2)

        self.conv_7 = tf.keras.layers.Conv2D(k, 7, 1, padding = 'valid', use_bias=False)
        self.batchnorm_7 = tf.keras.layers.BatchNormalization()

        self.layer_9 = BottleNeck(k)

        self.conv_8 = tf.keras.layers.Conv2D(3, 1, 1, padding = 'same', use_bias=False)

    def call(self, inputs):
        '''
        FULL NETWORK:
        out = tf.nn.leaky_relu(self.batchnorm_1(self.conv_1(inputs)), 0.1)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = tf.nn.leaky_relu(self.batchnorm_2(self.conv_2(out)), 0.1)
        out = self.layer_3(out)
        out = tf.nn.leaky_relu(self.batchnorm_3(self.conv_3(out)), 0.1)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = tf.nn.leaky_relu(self.batchnorm_4(self.conv_4(out)), 0.1)
        out = self.layer_6(out)
        out = tf.nn.leaky_relu(self.batchnorm_5(self.conv_5(out)), 0.1)
        out = self.layer_7(out)
        out = tf.nn.leaky_relu(self.batchnorm_6(self.conv_6(out)), 0.1)
        out = self.layer_8(out)
        out = tf.nn.leaky_relu(self.batchnorm_7(self.conv_7(out)), 0.1)
        out = self.layer_9(out)
        out = self.conv_8(out)
        out = tf.keras.activations.sigmoid(out)
        '''
        
        print(inputs.shape)
        #out = tf.nn.leaky_relu(self.batchnorm_1(self.conv_1(inputs)), 0.1)
        out = tf.nn.leaky_relu(self.conv_1(inputs), 0.1)
        print(out.shape)
        out = self.layer_1(out)
        #out = self.layer_2(out)
        out = tf.nn.leaky_relu(self.batchnorm_2(self.conv_2(out)), 0.1)
        #out = self.layer_3(out)
        #out = tf.nn.leaky_relu(self.batchnorm_3(self.conv_3(out)), 0.1)
        #out = self.layer_4(out)
        #out = self.layer_5(out)
        out = tf.nn.leaky_relu(self.batchnorm_4(self.conv_4(out)), 0.1)
        out = self.layer_6(out)
        #out = tf.nn.leaky_relu(self.batchnorm_5(self.conv_5(out)), 0.1)
        #out = self.layer_7(out)
        #out = tf.nn.leaky_relu(self.batchnorm_6(self.conv_6(out)), 0.1)
        #out = self.layer_8(out)
        out = tf.nn.leaky_relu(self.batchnorm_7(self.conv_7(out)), 0.1)
        #out = self.layer_9(out)
        out = self.conv_8(out)
        out = tf.keras.activations.sigmoid(out)
        return out
    
    def loss(self, labels, inputs):
        #Temporarily, juse use MSE loss
        #Later, this needs to be residual as defined in the paper
        return tf.keras.losses.mse(inputs, labels)