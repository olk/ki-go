import os
import tensorflow as tf
import time

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from kigo.data.processor import GoDataProcessor
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

    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name(), data_p=raw_p)
    generator = processor.load_go_data('train')
    print('number of train sample: %d' % generator.get_num_samples())
    test_generator = processor.load_go_data('test')

    input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
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
    model.fit_generator(generator=generator.generate(batch_size, num_classes),
                        epochs=epochs,
                        steps_per_epoch=generator.get_num_samples() / batch_size)
    elapsed = time.perf_counter() - start
    print('elapsed: {:0.3f}'.format(elapsed))

    model.evaluate_generator(generator=test_generator.generate(batch_size, num_classes),
                             steps=test_generator.get_num_samples() / batch_size)


if __name__ == '__main__':
    main()
