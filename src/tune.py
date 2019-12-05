import os
import kerastuner.tuners as kt
import kerastuner.engine.hyperparameters as kh
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from kigo.data.dataset_builder import SGFDataset
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.networks import betago


def build_hyperparams():
    hp = kh.HyperParameters()
    hp.Int('padding1', 2, 5, 1)
    hp.Int('conv_filters1', 8, 80, 8)
    hp.Int('kernel_size1', 2, 10, 1)
    hp.Int('padding2', 2, 5, 1)
    hp.Int('conv_filters2', 8, 80, 8)
    hp.Int('kernel_size2', 2, 10, 1)
    hp.Int('padding3', 2, 5, 1)
    hp.Int('conv_filters3', 8, 80, 8)
    hp.Int('kernel_size3', 2, 10, 1)
    hp.Int('padding4', 2, 5, 1)
    hp.Int('conv_filters4', 8, 80, 8)
    hp.Int('kernel_size4', 2, 10, 1)
    hp.Int('padding5', 2, 5, 1)
    hp.Int('conv_filters5', 8, 80, 8)
    hp.Int('kernel_size5', 2, 10, 1)
    hp.Int('padding6', 2, 5, 1)
    hp.Int('conv_filters6', 8, 80, 8)
    hp.Int('kernel_size6', 2, 10, 1)
    hp.Int('padding7', 2, 5, 1)
    hp.Int('conv_filters7', 8, 80, 8)
    hp.Int('kernel_size7', 2, 10, 1)
    hp.Choice('dense_units', [2**x for x in range(6, 11)])
    hp.Choice('activation', ['relu', 'elu', 'selu'])
    hp.Choice('learning_rate', [float(10**x) for x in range(-10, 1)])
    return hp

def build_model(hp, input_shape, classes):
    padding1 = hp['padding1']
    conv_filters1 = hp['conv_filters1']
    kernel_size1 = hp['kernel_size1']
    padding2 = hp['padding2']
    conv_filters2 = hp['conv_filters2']
    kernel_size2 = hp['kernel_size2']
    padding3 = hp['padding3']
    conv_filters3 = hp['conv_filters3']
    kernel_size3 = hp['kernel_size3']
    padding4 = hp['padding4']
    conv_filters4 = hp['conv_filters4']
    kernel_size4 = hp['kernel_size4']
    padding5 = hp['padding5']
    conv_filters5 = hp['conv_filters5']
    kernel_size5 = hp['kernel_size5']
    padding6 = hp['padding6']
    conv_filters6 = hp['conv_filters6']
    kernel_size6 = hp['kernel_size6']
    padding7 = hp['padding7']
    conv_filters7 = hp['conv_filters7']
    kernel_size7 = hp['kernel_size7']
    dense_units = hp['dense_units']
    activation = hp['activation']
    learning_rate = hp['learning_rate']
    network_layers = betago.layers(input_shape, classes,
                                   padding1, conv_filters1, kernel_size1,
                                   padding2, conv_filters2, kernel_size2,
                                   padding3, conv_filters3, kernel_size3,
                                   padding4, conv_filters4, kernel_size4,
                                   padding5, conv_filters5, kernel_size5,
                                   padding6, conv_filters6, kernel_size6,
                                   padding7, conv_filters7, kernel_size7,
                                   dense_units, activation)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

def main():
    # env
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    model_p = Path(os.environ.get('PATH_MODEL')).resolve()
    batch_size = int(os.environ.get('BATCH_SIZE'))
    sample_size = int(os.environ.get('SAMPLE_SIZE'))
    train_frac = float(os.environ.get('TRAIN_FRAC'))
    val_frac = float(os.environ.get('VAL_FRAC'))

    board_size = (19, 19)
    classes = 19 * 19
    shuffle_buffer = 1000
    prefetch_size = tf.data.experimental.AUTOTUNE

    encoder = SevenPlaneEncoder(board_size)

    builder = SGFDataset(encoder, train_frac, val_frac)
    ds_train = builder.as_dataset(split='train', as_supervised=True)
    ds_val = builder.as_dataset(split='validation', as_supervised=True)
    train_n = SGFDataset.size(ds_train)
    val_n = SGFDataset.size(ds_val)
    ds_train = ds_train.shuffle(shuffle_buffer)
    if 0 < sample_size:
        ds_train = ds_train.take(sample_size)
        train_n = sample_size
    ds_train = ds_train.repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)
    ds_val = ds_val.repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)

    input_shape = encoder.shape()
    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync

    hp = build_hyperparams()
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model(hp, input_shape, classes),
        objective='val_accuracy',
        hyperparameters=hp,
        max_epochs=15,
        factor=3,
        hyperband_iterations=3,
        distribution_strategy=strategy,
        directory=str(model_p),
        project_name='ki-go')
    tuner.search_space_summary()

    epochs=5
    tuner.search(ds_train,
                 validation_data=ds_val,
                 steps_per_epoch=train_n/epochs,
                 validation_steps=val_n/epochs,
                 epochs=epochs,
                 callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy')])

    tuner.results_summary(num_trials=2)


if __name__ == '__main__':
    main()
