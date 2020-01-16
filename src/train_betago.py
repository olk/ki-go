import mxnet as mx
import time

from pathlib import Path
from mxnet import autograd as ag
from mxnet.metric import Accuracy
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.utils import split_and_load

from kigo.data.dataset import SGFDatasetBuilder
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.networks.betago import Model

BATCH_SIZE = 1024
NUM_SEGMENTS = 500
EPOCHS = 25
GPU_COUNT = 2
PRINT_N = 5

def main():
    data_p = Path('/storage/data/ki-go').resolve()
    encoder = SevenPlaneEncoder((19, 19))
    builder = SGFDatasetBuilder(data_p, encoder=encoder)
    builder.download_and_prepare()
    train_ds = builder.as_train_dataset(batch_size=BATCH_SIZE, num_segments=NUM_SEGMENTS, shuffle=True)
    test_ds = builder.as_test_dataset(batch_size=BATCH_SIZE)
    # build nodel
    net = Model()
    # hybridize for speed
    net.hybridize(static_alloc=True, static_shape=True)
    # print graph
    shape = (1,) + encoder.shape()
    mx.viz.print_summary(
        net(mx.sym.var('data')),
        shape={'data':shape}
    )
    # pin GPUs
    ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    # optimizer
    opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
    opt = mx.optimizer.create('adam', **opt_params)
    # initialize parameters
    net.initialize(mx.init.Xavier(magnitude=2.3), ctx=ctx, force_reinit=True)
    # fetch and broadcast parameters
    params = net.collect_params()
    # trainer
    trainer = Trainer(params=params,
                      optimizer=opt,
                      kvstore='device')
    # loss function
    loss_fn = SoftmaxCrossEntropyLoss()
    # use accuracy as the evaluation metric
    metric = Accuracy()
    start = time.perf_counter()
    # train
    for train_itr in train_ds:
        start = time.perf_counter()
        for e in range(EPOCHS):
            tick = time.time()
            # reset the train data iterator.
            train_itr.reset()
            # loop over the train data iterator
            for i, batch in enumerate(train_itr):
                if i == 0:
                    tick_0 = time.time()
                # splits train data into multiple slices along batch_axis
                # copy each slice into a context
                data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                # splits train label into multiple slices along batch_axis
                # copy each slice into a context
                label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                outputs = []
                losses = []
                # inside training scope
                with ag.record():
                    for x, y in zip(data, label):
                        z = net(x)
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
                trainer.step(BATCH_SIZE)
                # updates internal evaluation
                metric.update(label, outputs)
                # Print batch metrics
                if 0 == i % PRINT_N and 0 < i:
                    print('Batch [{}], Accuracy {:.4f}, Samples/sec: {:.4f}'.format(
                        i, metric.get()[1], BATCH_SIZE*(PRINT_N)/(time.time()-tick))
                    )
                    tick = time.time()
            # gets the evaluation result
            print('Epoch [{}], Accuracy {:.4f}'.format(e, metric.get()[1]))
            print('~Samples/Sec {:.4f}'.format(BATCH_SIZE*(i+1)/(time.time()-tick_0)))
            # reset evaluation result to initial state
            metric.reset()

    elapsed = time.perf_counter() - start
    print('elapsed: {:0.3f}'.format(elapsed))
    # use Accuracy as the evaluation metric
    metric = Accuracy()
    for test_itr in test_ds:
        for batch in test_itr:
            data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(net(x))
            metric.update(label, outputs)
    print('validation %s=%f' % metric.get())

if __name__ == '__main__':
    main()
