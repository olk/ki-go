import mxnet as mx
import shutil

from pathlib import Path
from multiprocessing import cpu_count

from kigo.data.dataset import SGFDatasetBuilder
from kigo.callbacks.checkpointing import CheckpointingCallback
from kigo.callbacks.logging import LoggingCallback
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.networks import betago

GPU_COUNT = 2
BATCH_SIZE_PER_REPLICA = 1024
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * GPU_COUNT
NUM_GAMES = 10000
NUM_SEGMENTS = 100
NUM_PREFETCH = 4
EPOCHS = 200
PRINT_N = 100
LOOPS = 1

def main():
    data_p = Path('./data/').resolve()
    checkpoint_p = Path('./checkpoints/').resolve()
    checkpoint_p.mkdir(parents=True, exist_ok=True)
    checkpoint_p = checkpoint_p.joinpath('betago.params')
    logs_p = Path('./logs/').resolve()
    shutil.rmtree(logs_p, ignore_errors=True)
    # encoder
    encoder = SevenPlaneEncoder((19, 19))
    # generate data
    builder = SGFDatasetBuilder(data_p, encoder=encoder)
    builder.download_and_prepare()
    # two-GPU context
    ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    # build model
    model = betago.Model(encoder)
    # print graph
    model.summary()
    for _ in range(LOOPS):
        # training data
        train_itr = builder.train_iter(batch_size=BATCH_SIZE,
                                       max_workers=cpu_count(),
                                       num_games=NUM_GAMES,
                                       num_segments=NUM_SEGMENTS,
                                       num_prefetch=NUM_PREFETCH,
                                       shuffle=True)
        # fit model
        model.fit(train_itr,
                  ctx=ctx,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks=[
                      LoggingCallback(logs_p=logs_p, print_n=PRINT_N),
                      CheckpointingCallback(model, checkpoint_p=checkpoint_p)])

if __name__ == '__main__':
    main()
