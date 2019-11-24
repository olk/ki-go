from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout


def layers(input_shape, classes, filters, conv_kernel, pool_size):
    return [
        Conv2D(filters, conv_kernel, padding='valid', input_shape=input_shape, activation='relu', name='conv1'),
        Conv2D(filters, conv_kernel, activation='relu', name='conv2'),
        MaxPooling2D(pool_size=pool_size, name='pool'),
        Dropout(0.25, name='dropout1'),
        Flatten(name='flatten'),
        Dense(25, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout2'),
        Dense(classes, activation='relu', name='dense2'),
    ]
