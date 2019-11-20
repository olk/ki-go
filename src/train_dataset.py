import os
import tensorflow as tf
import time

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from kigo.data.processor import GoDataProcessor
from kigo.data.dataset_creator import SGFTFDatasetCreator
from kigo.encoders.oneplane import OnePlaneEncoder
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

    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    shuffle_buffer = 1000
    prefetch_size = tf.data.experimental.AUTOTUNE

    tfd_train = SGFTFDatasetCreator(processed_p.joinpath('train'), batch_size)
    ds_train = tfd_train.dataset
    tfd_val = SGFTFDatasetCreator(processed_p.joinpath('val'), batch_size, train=False)
    ds_val = tfd_val.dataset

    ds_train = ds_train.shuffle(shuffle_buffer).repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)
    ds_val = ds_val.repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)

    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))

    input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    #strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        network_layers = small.layers(input_shape)
        model = Sequential()
        for layer in network_layers:
            model.add(layer)
        model.add(Dense(num_classes, activation='softmax', name='YYY'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    checkpoints_p = model_p.joinpath('checkpoints')
    checkpoints_p.mkdir(exist_ok=True)
    start = time.perf_counter()
    model.fit(ds_train,
            validation_data=ds_val,
            steps_per_epoch=tfd_train.n / epochs,
            validation_steps=tfd_val.n / epochs,
            epochs=epochs,
            callbacks=[ModelCheckpoint(str(checkpoints_p.joinpath('small_model_epoch_{epoch}.h5')))])
    elapsed = time.perf_counter() - start
    print('elapsed: {:0.3f}'.format(elapsed))


if __name__ == '__main__':
    main()
