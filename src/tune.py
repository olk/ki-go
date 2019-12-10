import kerastuner.tuners as kt
import kerastuner.engine.hyperparameters as kh
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

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
    hp.Choice('dense_units', [2**x for x in range(6, 11)])
    hp.Choice('activation', ['relu', 'elu', 'selu']) # LeakyReLu, PreLu, ... https://keras.io/activations/?
    hp.Choice('learning_rate', [float(10**x) for x in range(-10, 1)])
    hp.Choice('batch_size', [2**x for x in range(4, 11)])
    return hp

def build_model(hp, input_shape, classes):
    dense_units = hp['dense_units']
    activation = hp['activation']
    learning_rate = hp['learning_rate']
    network_layers = betago.layers(input_shape, classes,
                                   64,
                                   64,
                                   64,
                                   48,
                                   48,
                                   32,
                                   32,
                                   7,
                                   5,
                                   5,
                                   5,
                                   5,
                                   5,
                                   5,
                                   dense_units, activation)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

def main():
    model_p = Path('model').resolve()
    train_frac = 0.8
    val_frac = 0.1
    sample_size = 100
    board_size = (19, 19)
    classes = 19 * 19
    shuffle_buffer = 1000
    prefetch_size = tf.data.experimental.AUTOTUNE

    hp = build_hyperparams()
    batch_size = hp['batch_size']

    encoder = SevenPlaneEncoder(board_size)

    builder = SGFDataset(encoder, train_frac, val_frac)
    ds_train = builder.as_dataset(split='train', as_supervised=True)
    ds_val = builder.as_dataset(split='validation', as_supervised=True)
    val_n = SGFDataset.size(ds_val)
    ds_train = ds_train.shuffle(shuffle_buffer)
    if 0 < sample_size:
        ds_train = ds_train.take(sample_size)
        train_n = sample_size
    else:
        train_n = SGFDataset.size(ds_train)
    ds_train = ds_train.repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)
    ds_val = ds_val.repeat().batch(batch_size).prefetch(buffer_size=prefetch_size)

    input_shape = encoder.shape()
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:1')
    #strategy = tf.distribute.MirroredStrategy()

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_model(hp, input_shape, classes),
        objective='val_accuracy',
        hyperparameters=hp,
        max_trials=25,
        distribution_strategy=strategy,
        directory=str(model_p),
        project_name='bayes')
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
