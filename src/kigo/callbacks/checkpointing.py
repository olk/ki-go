from kigo.callbacks.base import Callback

class CheckpointingCallback(Callback):
    def __init__(self, model, checkpoint_p, with_epoch=False):
        super().__init__()
        self._checkpoint_p = checkpoint_p
        self._model = model
        self._with_epoch=with_epoch

    def after_batch(self, epoch, batch, batch_size, metric):
        pass

    def before_epoch(self, epoch):
        pass

    def after_epoch(self, epoch, batch, batch_size, metric):
        if self._with_epoch:
            self._model.save(self._checkpoint_p % epoch)
        else:
            self._model.save(self._checkpoint_p)
