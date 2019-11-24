import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import time

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from kigo.data.dataset_builder import SGFDataset
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.networks import small


def main():
    # env
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    raw_p = Path(os.environ.get('PATH_RAW')).resolve()
    processed_p = Path(os.environ.get('PATH_PROCESSED')).resolve()
    model_p = Path(os.environ.get('PATH_MODEL')).resolve()
    batch_size = int(os.environ.get('BATCH_SIZE'))
    epochs = int(os.environ.get('EPOCHS'))
    sample_size = int(os.environ.get('SAMPLE_SIZE'))
    train_frac = float(os.environ.get('TRAIN_FRAC'))
    val_frac = float(os.environ.get('VAL_FRAC'))
    lr = 1e-6

    board_size = (19, 19)
    classes = 19 * 19
    shuffle_buffer = 1000
    prefetch_size = tf.data.experimental.AUTOTUNE
    filters = 32
    kernel_size = 3
    pool_size = 2
    drop_out1 = 0.25
    drop_out2 = 0.5
    dense_units = 25

    encoder = SevenPlaneEncoder(board_size)

    builder = SGFDataset(encoder, train_frac, val_frac)
    ds_train = builder.as_dataset(split='train', as_supervised=True)
    ds_val = builder.as_dataset(split='validation', as_supervised=True)
    train_n = SGFDataset.size(ds_train)
    val_n = SGFDataset.size(ds_val)
    ds_train = ds_train.shuffle(shuffle_buffer).repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)
    if 0 < sample_size:
        ds_train = ds_train.take(sample_size)
    ds_val = ds_val.repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)

    input_shape = encoder.shape()
    #strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        network_layers = small.layers(input_shape=input_shape,
                                      classes=classes,
                                      filters=filters,
                                      kernel_size=kernel_size,
                                      pool_size=pool_size,
                                      drop_out1=drop_out1,
                                      drop_out2=drop_out2,
                                      dense_units=dense_units)
        model = Sequential()
        for layer in network_layers:
            model.add(layer)
        model.summary()
        # initialize optimizer
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    checkpoints_p = model_p.joinpath('checkpoints')
    checkpoints_p.mkdir(exist_ok=True)
    start = time.perf_counter()
    print('total dataset items: %d' % train_n)
    print('used dataset items: %d' % (train_n if 0 == sample_size else sample_size))
    model.fit(ds_train,
            validation_data=ds_val,
            steps_per_epoch=train_n / epochs,
            validation_steps=val_n / epochs,
            epochs=epochs,
            callbacks=[
                EarlyStopping(patience=2, monitor='val_loss'),
                ModelCheckpoint(str(checkpoints_p.joinpath('small_model_epoch_{epoch}.h5')))
            ])
    elapsed = time.perf_counter() - start
    print('elapsed: {:0.3f}'.format(elapsed))


if __name__ == '__main__':
    main()
