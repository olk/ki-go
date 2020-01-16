from mxnet.gluon.nn import Conv2D, Dense, Flatten, HybridBlock

# based on https://github.com/maxpumperla/betago


class Model(HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = Conv2D(64, kernel_size=(7, 7), padding=(3, 3))
            self.conv2 = Conv2D(64, kernel_size=(5, 5), padding=(2, 2))
            self.conv3 = Conv2D(64, kernel_size=(5, 5), padding=(2, 2))
            self.conv4 = Conv2D(48, kernel_size=(5, 5), padding=(2, 2))
            self.conv5 = Conv2D(48, kernel_size=(5, 5), padding=(2, 2))
            self.conv6 = Conv2D(32, kernel_size=(5, 5), padding=(2, 2))
            self.conv7 = Conv2D(32, kernel_size=(5, 5), padding=(2, 2))
            self.flatten = Flatten()
            self.dense1 = Dense(1024)
            self.dense2 = Dense(19 * 19)

    def hybrid_forward(self, F, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
