import mxnet as mx
import mxboard as mxb
import shutil
import time

from pathlib import Path
from multiprocessing import cpu_count
from mxnet import autograd as ag
from mxnet.metric import Accuracy
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.utils import split_and_load
from mxnet import profiler

from kigo.data.dataset import SGFDatasetBuilder
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.networks.betago import Model

GPU_COUNT = 2
BATCH_SIZE_PER_REPLICA = 512
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * GPU_COUNT
EPOCHS = 25
PRINT_N = 50
FACTOR = 20

def main():
    data_p = Path('/storage/data/').resolve()
    checkpoint_p = Path('./checkpoints/').resolve()
    checkpoint_p.mkdir(parents=True, exist_ok=True)
    logs_p = Path('./logs/').resolve()
    shutil.rmtree(logs_p, ignore_errors=True)
    encoder = SevenPlaneEncoder((19, 19))
    builder = SGFDatasetBuilder(data_p, encoder=encoder)
    builder.download_and_prepare()
    train_itr = builder.train_dataset(batch_size=BATCH_SIZE, max_worker=cpu_count(), factor=FACTOR)
    test_itr = builder.test_dataset(batch_size=BATCH_SIZE, max_worker=cpu_count(), factor=FACTOR)
    # build model
    betago = Model()
    # convert to half-presicion floating point FP16
    # NOTE: all NVIDIA GPUs with compute capability 6.1 have a low-rate FP16 performance == FFP16 is not the fast path on these GPUs
    #       data passed to split_and_load() must be float16 too
    #betago.cast('float16')
    # hybridize for speed
    betago.hybridize(static_alloc=True, static_shape=True)
    # print graph
    shape = (1,) + encoder.shape()
    mx.viz.print_summary(
        betago(mx.sym.var('data')),
        shape={'data':shape}
    )
    # pin GPUs
    ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    # optimizer
    opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
    opt = mx.optimizer.create('adam', **opt_params)
    # initialize parameters
    # MXNet initializes the weight matrices uniformly by drawing from [−0.07,0.07], bias parameters are all set to 0
    # 'Xavier': initializer is designed to keep the scale of gradients roughly the same in all layers
    betago.initialize(mx.init.Xavier(magnitude=2.3), ctx=ctx, force_reinit=True)
    # fetch and broadcast parameters
    params = betago.collect_params()
    # trainer
    trainer = Trainer(params=params,
                      optimizer=opt,
                      kvstore='device')
    # loss function
    loss_fn = SoftmaxCrossEntropyLoss()
    # use accuracy as the evaluation metric
    metric = Accuracy()
    with mxb.SummaryWriter(logdir='./logs') as sw:
        # add graph to MXBoard
        #betago.forward(mx.nd.ones(shape, ctx=ctx[0]))
        #betago.forward(mx.nd.ones(shape, ctx=ctx[1]))
        #sw.add_graph(betago)
        profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')
        start = time.perf_counter()
        # train
        for e in range(EPOCHS):
            if 0 == e:
                profiler.set_state('run')
            tick = time.time()
            # reset the train data iterator.
            train_itr.reset()
            # loop over the train data iterator
            for i, batch in enumerate(train_itr):
                if 0 == i:
                    tick_0 = time.time()
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
                        z = betago(x)
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
                    # checkpointing
                    betago.save_parameters(str(checkpoint_p.joinpath('betago-{}.params'.format(e))))
                    sw.add_scalar(tag='Accuracy', value={'naive':metric.get()[1]}, global_step=i-PRINT_N)
                    sw.add_scalar(tag='Speed', value={'naive':BATCH_SIZE*(PRINT_N)/(time.time()-tick)}, global_step=i-PRINT_N)
                    print('epoch[{}] batch [{}], accuracy {:.4f}, samples/sec: {:.4f}'.format(
                        e, i, metric.get()[1], BATCH_SIZE*(PRINT_N)/(time.time()-tick))
                    )
                    tick = time.time()
            if 0 == e:
                profiler.set_state('stop')
                profiler.dump()
            # gets the evaluation result
            print('epoch [{}], accuracy {:.4f}, samples/sec: {:.4f}'.format(
                e, metric.get()[1], BATCH_SIZE*(i+1)/(time.time()-tick_0))
            )
            # reset evaluation result to initial state
            metric.reset()

    elapsed = time.perf_counter() - start
    print('elapsed: {:0.3f}'.format(elapsed))
    # use Accuracy as the evaluation metric
    metric = Accuracy()
    for batch in test_itr:
        data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(betago(x))
        metric.update(label, outputs)
    print('validation %s=%f' % metric.get())

if __name__ == '__main__':
    main()
