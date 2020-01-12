import mxnet as mx
import time

from pathlib import Path
from mxnet import autograd as ag
from mxnet.metric import Accuracy
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.utils import split_and_load

from kigo.data.dataset import SGFDatasetBuilder
from kigo.networks.betago import Model

BATCH_SIZE = 1024
NUM_GAMES = 10000
EPOCHS = 10
GPU_COUNT = 2

def main():
    data_p = Path('/storage/data/ki-go').resolve()
    builder = SGFDatasetBuilder( data_p)
    builder.download_and_prepare()
    #train_data = builder.get_train_iter(BATCH_SIZE, shuffle=True, num_games=NUM_GAMES)
    #test_data = builder.get_test_iter(BATCH_SIZE, num_games=NUM_GAMES)
    train_data = builder.get_train_array(num_games=NUM_GAMES)
    train_data = mx.io.NDArrayIter(train_data[0], train_data[1], BATCH_SIZE)
    #test_data = builder.get_test_array(num_games=NUM_GAMES)
    #test_data = mx.io.NDArrayIter(test_data[0], test_data[1], BATCH_SIZE)
    # build nodel
    net = Model()
    # hybridize for speed
    net.hybridize(static_alloc=True, static_shape=True)
    # pin GPUs
    ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    # optimizer
    opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
    opt = mx.optimizer.create('adam', **opt_params)
    # initialize parameters
    net.initialize(force_reinit=True, ctx=ctx)
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
    for epoch in range(1, EPOCHS+1):
        # reset the train data iterator.
        train_data.reset()
        # reset evaluation result to initial state
        metric.reset()
        # loop over the train data iterator
        for step, batch in enumerate(train_data, 1):
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
            name, acc = metric.get()
            print('Step %d of Epoch %d: training %s=%f' % (step, epoch, name, acc))
        # gets the evaluation result
        name, acc = metric.get()
        print('Epoch %d/%d: training %s=%f' % (epoch, EPOCHS, name, acc))
    # use Accuracy as the evaluation metric
    #metric = Accuracy()
    #for batch in test_data:
    #    data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    #    label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    #    outputs = []
    #    for x in data:
    #        outputs.append(net(x))
    #    metric.update(label, outputs)
    #print('validation %s=%f' % metric.get())
    elapsed = time.perf_counter() - start
    print('elapsed: {:0.3f}'.format(elapsed))

if __name__ == '__main__':
    main()
