import abc


class Callback(abc.ABC):
    @abc.abstractmethod
    def after_batch(self, epoch, batch, batch_size, metric):
        "batch computation finished"

    @abc.abstractmethod
    def before_epoch(self, epoch):
        "epoch computation started"

    @abc.abstractmethod
    def after_epoch(self, epoch, batch, batch_size, metric):
        "epoch computation inished"

