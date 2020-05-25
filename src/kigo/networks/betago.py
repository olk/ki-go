import mxnet as mx

from mxnet import autograd as ag
from mxnet.gluon import Trainer
from mxnet.gluon.nn import Conv2D, Dense, Flatten, HybridBlock
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.utils import split_and_load
from mxnet.gluon.metric import Accuracy

# based on https://github.com/maxpumperla/self._net


class Model:
    class Net(HybridBlock):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            with self.name_scope():
                self.conv1 = Conv2D(64, kernel_size=(7, 7), padding=(3, 3))
                self.conv2 = Conv2D(64, kernel_size=(5, 5), padding=(2, 2))
                self.conv3 = Conv2D(64, kernel_size=(5, 5), padding=(2, 2))
                self.conv4 = Conv2D(64, kernel_size=(5, 5), padding=(2, 2))
                self.conv5 = Conv2D(48, kernel_size=(5, 5), padding=(2, 2))
                self.conv6 = Conv2D(48, kernel_size=(5, 5), padding=(2, 2))
                self.conv7 = Conv2D(48, kernel_size=(5, 5), padding=(2, 2))
                self.conv8 = Conv2D(32, kernel_size=(5, 5), padding=(2, 2))
                self.conv9 = Conv2D(32, kernel_size=(5, 5), padding=(2, 2))
                self.conv10 = Conv2D(32, kernel_size=(5, 5), padding=(2, 2))
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
            x = F.relu(self.conv8(x))
            x = F.relu(self.conv9(x))
            x = F.relu(self.conv10(x))
            x = self.flatten(x)
            x = F.relu(self.dense1(x))
            #x = F.softmax(self.dense2(x))
            x = self.dense2(x)
            return x

    def __init__(self, encoder, checkpoint_p=None):
        self._encoder = encoder
        self._net = Model.Net()
        # convert to half-presicion floating point FP16
        # NOTE: all NVIDIA GPUs with compute capability 6.1 have a low-rate FP16 performance == FFP16 is not the fast path on these GPUs
        #       data passed to split_and_load() must be float16 too
        #self._net.cast('float16')
        # hybridize for speed
        self._net.hybridize(static_alloc=True, static_shape=True)

    @classmethod
    def create(cls, encoder, checkpoint_p, ctx):
        assert checkpoint_p is not None
        assert checkpoint_p.exists()
        assert checkpoint_p.is_file()
        model = Model(encoder)
        # load parameters from checkpoint
        # initialize parameters
        model._net.initialize(mx.init.Xavier(magnitude=2.3), ctx=ctx, force_reinit=True)
        model._net.load_parameters(str(checkpoint_p))
        return model

    def summary(self):
        # print graph
        shape = (1,) + self._encoder.shape()
        mx.viz.print_summary(
            self._net(mx.sym.var('data')),
            shape={'data':shape}
        )

    def save(self, checkpoint_p):
        self._net.save_parameters(str(checkpoint_p))

    def fit(self, itr, ctx, epochs, batch_size, callbacks=None):
        # ADAM optimizer
        #opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
        opt = mx.optimizer.create('adam')
        # SGD optimizer 
        #opt = mx.optimizer.create('sgd')
        # AdaDelta optimizer 
        #opt = mx.optimizer.create('adadelta')
        # initialize parameters
        # MXNet initializes the weight matrices uniformly by drawing from [−0.07,0.07], bias parameters are all set to 0
        # 'Xavier': initializer is designed to keep the scale of gradients roughly the same in all layers
        self._net.initialize(mx.init.Xavier(magnitude=2.3), ctx=ctx, force_reinit=True)
        # fetch and broadcast parameters
        params = self._net.collect_params()
        # trainer
        trainer = Trainer(params=params,
                          optimizer=opt,
                          kvstore='device')
        # loss function
        loss_fn = SoftmaxCrossEntropyLoss()
        # use accuracy as the evaluation metric
        metric = Accuracy()
        # train
        for e in range(epochs):
            if callbacks is not None:
                for cb in callbacks:
                    cb.before_epoch(e)
            # reset evaluation result to initial state
            metric.reset()
            # reset the train data iterator.
            itr.reset()
            # loop over the train data iterator
            for i, batch in enumerate(itr):
                # splits train data into multiple slices along batch_axis
                # copy each slice into a context
                data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
                # splits train label into multiple slices along batch_axis
                # copy each slice into a context
                label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
                outputs = []
                losses = []
                # inside training scope
                with ag.record():
                    for x, y in zip(data, label):
                        z = self._net(x)
                        # computes softmax cross entropy loss
                        l = loss_fn(z, y)
                        outputs.append(z)
                        losses.append(l)
                # backpropagate the error for one iteration
                for l in losses:
                    l.backward()
                # make one step of parameter update.
                # trainer needs to know the batch size of data
                # to normalize the gradient by 1/batch_size
                trainer.step(batch_size)
                # updates internal evaluation
                metric.update(label, outputs)
                # invoke callbacks after batch
                if callbacks is not None:
                    for cb in callbacks:
                        cb.after_batch(e, i, batch_size, metric)
            # invoke callbacks after epoch
            if callbacks is not None:
                for cb in callbacks:
                    cb.after_epoch(e, i, batch_size, metric)
        return metric

    def evaluate(self, itr, ctx):
        metric = Accuracy()
        itr.reset()
        for batch in itr:
            data = batch.data[0].as_in_context(context=ctx)
            label = batch.label[0].as_in_context(context=ctx)
            output = self._net(data)
            metric.update(label, output)
        return metric

    def predict(self, x):
        prediction = self._net(x)
        # wait_to_read is essentially a blocking call to synchronize the execution
        # it ensures that the result of the computation to produce an ndarray is completed
        # by the backend engine
        prediction.wait_to_read()
        return prediction
