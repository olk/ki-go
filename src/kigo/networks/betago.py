from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Layer

# based on https://github.com/maxpumperla/betago

def layers(input_shape, classes,
           padding1, conv_filters1, kernel_size1,
           padding2, conv_filters2, kernel_size2,
           padding3, conv_filters3, kernel_size3,
           padding4, conv_filters4, kernel_size4,
           padding5, conv_filters5, kernel_size5,
           padding6, conv_filters6, kernel_size6,
           padding7, conv_filters7, kernel_size7,
           dense_units, activation):
    return [
        ZeroPadding2D((padding1, padding1), input_shape=input_shape, name='zero-padding1'),
        Conv2D(conv_filters1, (kernel_size1, kernel_size1), padding='valid', activation=activation, name='conv1'),
        ZeroPadding2D((padding2, padding2), name='zero-padding2'),
        Conv2D(conv_filters2, (kernel_size2, kernel_size2), padding='valid', activation=activation, name='conv2'),
        ZeroPadding2D((padding3, padding3), name='zero-padding3'),
        Conv2D(conv_filters3, (kernel_size3, kernel_size3), padding='valid', activation=activation, name='conv3'),
        ZeroPadding2D((padding4, padding4), name='zero-padding4'),
        Conv2D(conv_filters4, (kernel_size4, kernel_size4), padding='valid', activation=activation, name='conv4'),
        ZeroPadding2D((padding5, padding5), name='zero-padding5'),
        Conv2D(conv_filters5, (kernel_size5, kernel_size5), padding='valid', activation=activation, name='conv5'),
        ZeroPadding2D((padding6, padding6), name='zero-padding6'),
        Conv2D(conv_filters6, (kernel_size6, kernel_size6), padding='valid', activation=activation, name='conv6'),
        ZeroPadding2D((padding7, padding7), name='zero-padding7'),
        Conv2D(conv_filters7, (kernel_size7, kernel_size7), padding='valid', activation=activation, name='conv7'),
        Flatten(name='flatten'),
        Dense(units=dense_units, activation=activation, name='dense1'),
        Dense(units=classes, activation='softmax', name='dense2'),
    ]