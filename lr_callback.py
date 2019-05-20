import numpy as np
import keras
import itertools
from keras import backend as K

class CyclicCosineLR(keras.callbacks.Callback):
    def __init__(self, lr_min=0.001, lr_max=0.4, number_of_batches=120):
        super(CyclicCosineLR, self).__init__()
        self.number_of_batches = number_of_batches
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.cosine_lr()

    def on_train_begin(self, logs={}):
        self.update_lr()

    def cosine_lr(self):
        T_current = np.arange(self.number_of_batches)
        lr_current = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos((T_current / (self.number_of_batches)) * np.pi))
                
        self.lr_range_iter = itertools.cycle(lr_current.tolist())

    def update_lr(self):
        K.set_value(self.model.optimizer.lr, next(self.lr_range_iter))

    def on_batch_end(self, batch, logs={}):
        self.update_lr()