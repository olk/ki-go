import mxboard as mxb
import time

from kigo.callbacks.base import Callback

class LoggingCallback(Callback):
    def __init__(self, logs_p, print_n=100):
        super().__init__()
        self._print_n = print_n
        self._sw =  mxb.SummaryWriter(logdir=str(logs_p))

    def after_batch(self, epoch, batch, batch_size, metric):
        # print batch metrics
        if 0 == batch % self._print_n and 0 < batch:
            self._sw.add_scalar(tag='Accuracy', value={'naive':metric.get()[1]}, global_step=batch-self._print_n)
            self._sw.add_scalar(tag='Speed', value={'naive':batch_size*(self._print_n)/(time.time()-self._tick)}, global_step=batch-self._print_n)
            print('epoch[{}] batch [{}], accuracy {:.4f}, samples/sec: {:.1f}'.format(
                epoch, batch, metric.get()[1], batch_size*(self._print_n)/(time.time()-self._tick))
            )
            self._tick = time.time()

    def before_epoch(self, epoch):
        self._tick = time.time()
        self._tick0 = time.time()

    def after_epoch(self, epoch, batch, batch_size, metric):
        # print the evaluation result
        print('epoch [{}], accuracy {:.4f}, samples/sec: {:.1f}'.format(
            epoch, metric.get()[1], batch_size*(batch+1)/(time.time()-self._tick0))
        )
