import mxnet as mx
import shutil

from pathlib import Path
from multiprocessing import cpu_count

from kigo.data.dataset import SGFDatasetBuilder
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.networks import betago

GPU_COUNT = 1
BATCH_SIZE_PER_REPLICA = 1024
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * GPU_COUNT
NUM_GAMES = 100
NUM_SEGMENTS = 1
NUM_PREFETCH = 4

def main():
    data_p = Path('./data/').resolve()
    checkpoint_p = Path('./checkpoints/').resolve().joinpath('betago.params')
    # encoder
    encoder = SevenPlaneEncoder((19, 19))
    # generate data
    builder = SGFDatasetBuilder(data_p, encoder=encoder)
    builder.download_and_prepare()
    # GPU context
    ctx = mx.gpu(1)
    # build model
    model = betago.Model.create(encoder, checkpoint_p=checkpoint_p, ctx=ctx)
    # print graph
    model.summary()
    # test data
    test_itr = builder.test_iter(batch_size=BATCH_SIZE,
                                 max_workers=cpu_count(),
                                 num_games=NUM_GAMES,
                                 num_segments=NUM_SEGMENTS,
                                 num_prefetch=NUM_PREFETCH)
    # evaluate model
    metric = model.evaluate(test_itr, ctx=ctx)
    print('validation %s=%f' % metric.get())

if __name__ == '__main__':
    main()
