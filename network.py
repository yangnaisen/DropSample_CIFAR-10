from datetime import datetime

import keras
import numpy as np
from keras import backend as K
from keras import layers
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.regularizers import l2
from batchnorm_keras import BatchNormalizationCustom
import batchnorm_keras
from save_record import save_record

pytorch_initializers = VarianceScaling(
    scale=1. / 2000., mode='fan_in', distribution='uniform', seed=None)

weight_decay = 5e-4 * 512


class CallBackModel(Model):
    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        self._check_trainable_weights_consistency()
        if self.train_function is None:
            inputs = (self._feed_inputs + self._feed_targets +
                      self._feed_sample_weights)
            if self._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]

            with K.name_scope('training'):
                with K.name_scope(self.optimizer.__class__.__name__):
                    training_updates = self.optimizer.get_updates(
                        params=self._collected_trainable_weights,
                        loss=self.total_loss)
                updates = (
                    self.updates + training_updates + self.metrics_updates)
                # Gets loss and metrics. Updates weights at each call.
                self.train_function = K.function(
                    inputs,
                    [self.total_loss] + self.metrics_tensors + self.outputs,
                    updates=updates,
                    name='train_function',
                    **self._function_kwargs)

    def train_dropsample(self,
                         number_epoch,
                         x_train,
                         x_train_raw,
                         y_train,
                         x_test,
                         y_test,
                         data_loader,
                         batch_size,
                         threshold_prob,
                         threshold_portion,
                         record_file_name,
                         shuffle=True):
        num_train_samples = x_train.shape[0]
        num_test_samples = x_test.shape[0]

        selected_train_index = np.arange(num_train_samples)
        dropped_train_index = np.array([], dtype=np.int64)

        train_start_time = datetime.now()

        y_train_int_label = y_train.copy().argmax(axis=1)
        y_test_int_label = y_test.copy().argmax(axis=1)

        train_record_list = []

        print_title = [
            'epoch', 'train time', 'train acc', 'test time', 'test acc',
            'dropped samples', 'total time'
        ]
        print(*(f'{item:>16s}' for item in print_title))

        for e in range(number_epoch):
            epoch_train_start_time = datetime.now()

            if shuffle:
                np.random.shuffle(selected_train_index)

            part_batch = selected_train_index.shape[0] % batch_size

            if part_batch != 0:
                part_batch_index = selected_train_index[-part_batch:].copy()
                selected_train_index = selected_train_index[
                    0:-part_batch].copy()
                dropped_train_index = np.concatenate(
                    [part_batch_index, dropped_train_index], axis=0)

            selected_x_train = x_train[selected_train_index].copy()
            selected_y_train = y_train[selected_train_index].copy()

            if len(dropped_train_index) > 0:
                dropped_x_train = x_train_raw[dropped_train_index].copy()
                dropped_y_train = y_train[dropped_train_index].copy()

            train_generator = data_loader(
                selected_x_train,
                selected_y_train,
                batch_size=batch_size,
                crop_size=32,
                cutout_size=8,
                is_train=True)

            history = GetPrediction()

            self.fit_generator(
                train_generator,
                epochs=1,
                workers=1,
                verbose=0,
                use_multiprocessing=False,
                validation_data=None,
                max_queue_size=10,
                shuffle=False,
                callbacks=[history])

            selected_prediction = history.prediction.copy()

            batchnorm_keras.DROP_SAMPLE = True
            dropped_prediction = self.predict(
                dropped_x_train, batch_size=batch_size).copy()
            batchnorm_keras.DROP_SAMPLE = False

            train_prediction = np.concatenate(
                [selected_prediction, dropped_prediction], axis=0)
            train_prediction_index = np.concatenate(
                [selected_train_index, dropped_train_index], axis=0)

            max_position = y_train_int_label[train_prediction_index]
            sequence_index = np.arange(num_train_samples)
            prob_error = 1.0 - train_prediction[(sequence_index, max_position)]

            selected_index_mask = (prob_error > threshold_prob)
            dropped_index_mask = (prob_error <= threshold_prob)

            selected_train_index = train_prediction_index[selected_index_mask]
            dropped_train_index = train_prediction_index[dropped_index_mask]

            dropped_portion = dropped_train_index.shape[0] / x_train.shape[0]

            if dropped_portion > threshold_portion:
                threshold_prob = threshold_prob / 10
                if threshold_portion < 0.9:
                    threshold_portion = threshold_portion + 0.21
                else:
                    threshold_portion = threshold_portion + 0.021

            train_accuracy = (train_prediction.argmax(
                axis=1) == max_position).sum() / num_train_samples

            epoch_train_end_time = datetime.now()
            epoch_training_time = (
                epoch_train_end_time - epoch_train_start_time).seconds

            test_prediction = self.predict(
                x_test, batch_size=batch_size).copy()
            test_accuracy = (test_prediction.argmax(
                axis=1) == y_test_int_label).sum() / num_test_samples

            epoch_test_time = (datetime.now() - epoch_train_end_time).seconds
            total_time = (datetime.now() - train_start_time).seconds

            print_record = [
                e + 1, epoch_training_time, train_accuracy, epoch_test_time,
                test_accuracy,
                len(dropped_train_index), total_time
            ]
            train_record_list.append(print_record)
            if e % 10 == 0:
                save_record(record_file_name, train_record_list)
            print(*(
                f'{item:16.4f}' if isinstance(item, np.float) else f'{item:16}'
                for item in print_record))


class GetPrediction(keras.callbacks.Callback):
    def __init__(self):
        super(GetPrediction, self).__init__()
        self.batch_predictions = []

    def on_batch_end(self, batch, logs={}):
        self.batch_predictions.append(logs.get('predictions').copy())

    def on_epoch_begin(self, epoch, logs=None):
        self.model.metrics_names.append('predictions')

    def on_epoch_end(self, epoch, logs=None):
        self.prediction = np.concatenate(self.batch_predictions, axis=0)


def conv_bn(x, number_filters, name):
    x = layers.Conv2D(
        number_filters, (3, 3),
        padding='same',
        use_bias=False,
        name=name + '_conv',
        kernel_regularizer=l2(weight_decay),
        kernel_initializer=pytorch_initializers)(x)

    x = BatchNormalizationCustom(
        axis=-1,
        momentum=0.9,
        epsilon=1e-5,
        #gamma_regularizer = l2(weight_decay),
        name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    return x


def residual(x, number_filters, name):
    res1 = conv_bn(x, number_filters, name + '_res1_')
    res2 = conv_bn(res1, number_filters, name + '_res2_')
    add = layers.Add()([res2, x])
    return add


def build_cifar_10_res_net():
    factor_filters = 2
    num_classes = 10
    input_image = layers.Input(shape=(32, 32, 3))
    prep = conv_bn(input_image, 32 * factor_filters, 'prep')

    layer1_conv = conv_bn(prep, 64 * factor_filters, 'layer1_conv_')
    layer1_pool = layers.MaxPooling2D(
        2, strides=2, name='layer1_pool')(layer1_conv)

    layer1_res = residual(layer1_pool, 64 * factor_filters, 'layer1_residual')

    layer2_conv = conv_bn(layer1_res, 128 * factor_filters, 'layer2_conv_')

    layer2_pool = layers.MaxPooling2D(
        2, strides=2, name='layer2_pool')(layer2_conv)

    layer3_conv = conv_bn(layer2_pool, 256 * factor_filters, 'layer3_conv_')

    layer3_pool = layers.MaxPooling2D(
        2, strides=2, name='layer3_pool')(layer3_conv)
    layer3_res = residual(layer3_pool, 256 * factor_filters, 'layer3_residual')

    final_pool = layers.MaxPooling2D(
        4, strides=4, name='final_pool')(layer3_res)
    flat_pool = layers.Flatten()(final_pool)
    logits = layers.Dense(
        num_classes,
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer=pytorch_initializers)(flat_pool)
    weighted_logits = layers.Lambda(lambda x: x * 0.125)(logits)

    prediction_scores = layers.Activation('softmax')(weighted_logits)
    model = CallBackModel(inputs=[input_image], outputs=[prediction_scores])

    return model
