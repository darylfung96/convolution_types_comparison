import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout


class StandardConv:
    def __init__(self, input_shape):
        self.current_epoch = 0
        self.model = self._build_model(input_shape)
        self.model_name = self.__class__.__name__
        self._build_summary()

    def _build_summary(self):
        self.tensorboard_loss = tf.placeholder(dtype=tf.float32)
        self.tensorboard_train_acc = tf.placeholder(dtype=tf.float32)
        self.tensorboard_val_acc = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar('loss', self.tensorboard_loss)
        tf.summary.scalar('training accuracy', self.tensorboard_train_acc)
        tf.summary.scalar('validation accuracy', self.tensorboard_val_acc)
        self.merged = tf.summary.merge_all()

        self.file_writer = tf.summary.FileWriter('logs/%s_summary' % self.model_name)
        self.summary_sess = tf.Session()

        # -- write the total parameters to tensorboard -- #
        self.tensorboard_total_parameters = tf.placeholder(dtype=tf.int32)
        total_param_scalar = tf.summary.scalar('total parameters', self.tensorboard_total_parameters)
        total_param_merge = tf.summary.merge([total_param_scalar])

        total_param_summary = self.summary_sess.run(total_param_merge,
                                                    feed_dict={
                                                        self.tensorboard_total_parameters: self.model.count_params()
                                                    })
        self.file_writer.add_summary(total_param_summary, 0)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, 3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, 5, strides=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, 5, strides=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def fit(self, x, y, validation_data=None, epochs=5, *args, **kwargs):
        for _ in range(epochs):
            self.model.fit(x, y, initial_epoch=self.current_epoch, epochs=self.current_epoch+1, validation_data=validation_data, *args, **kwargs)

            val_loss, val_accuracy = self.model.evaluate(validation_data[0], validation_data[1], 32)
            train_loss, train_accuracy = self.model.evaluate(x[:10000], y[:10000], 32)

            summary = self.summary_sess.run(self.merged, feed_dict={self.tensorboard_loss: val_loss,
                                                                    self.tensorboard_train_acc: train_accuracy,
                                                                    self.tensorboard_val_acc: val_accuracy})
            self.file_writer.add_summary(summary, self.current_epoch)
            self.current_epoch += 1



