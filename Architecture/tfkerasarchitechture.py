import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Lambda


def lrn_layer(x):
    # N = 5, depth_radius = (n-1)/2 = 2, N is the number of channels
    return tf.nn.local_response_normalization(x, depth_radius=2, bias=2.0, alpha=1e-4, beta=0.75)

def AlexNet(input_shape, num_classes):

    alexnet = Sequential(
        [
            Input(shape=input_shape),


            Conv2D(filters = 96, kernel_size = (11,11), strides = (4,4), activation = "relu"),

            Lambda(lrn_layer),
            
            MaxPooling2D(pool_size = (3,3), strides = (2,2)),


            Conv2D(filters = 256, kernel_size = (5,5), padding = 2, strides = (1,1), activation = "relu"),

            Lambda(lrn_layer),

            MaxPooling2D(pool_size = (3,3), strides = (2,2)),


            Conv2D(filters = 384, kernel_size = (3,3), padding = 1, strides = (1,1), activation = "relu"),


            Conv2D(filters = 384, kernel_size = (3,3), padding = 1, strides = (1,1), activation = "relu"),


            Conv2D(filters = 256, kernel_size = (3,3), padding = 1, strides = (1,1), activation = "relu"),
            MaxPooling2D(pool_size = (3,3), strides = (2,2)),


            Flatten(),

            Dense(units = 4096, activation= "relu"),

            Dropout(rate = 0.5),

            Dense(units = 4096, activation= "relu"),

            Dropout(rate = 0.5),

            Dense(units = num_classes, activation = "softmax"),


        ]
    )

    return alexnet

    

