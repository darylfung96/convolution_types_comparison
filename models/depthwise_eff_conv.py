from keras.layers import BatchNormalization, Flatten, Dense, SeparableConv2D, DepthwiseConv2D, Dropout
from keras.models import Sequential

from models.standard_conv import StandardConv


class DepthWiseEffConv(StandardConv):
    def __init__(self, input_shape):
        super(DepthWiseEffConv, self).__init__(input_shape)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(SeparableConv2D(32, (3, 1), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(DepthwiseConv2D((1, 3), activation='relu'))
        model.add(BatchNormalization())

        model.add(SeparableConv2D(32, (3, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(DepthwiseConv2D((1, 3), activation='relu'))
        model.add(BatchNormalization())

        model.add(SeparableConv2D(32, (5, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(DepthwiseConv2D((1, 5), strides=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(SeparableConv2D(64, (3, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(DepthwiseConv2D((1, 3), activation='relu'))
        model.add(BatchNormalization())

        model.add(SeparableConv2D(64, (3, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(DepthwiseConv2D((1, 3), activation='relu'))
        model.add(BatchNormalization())

        model.add(SeparableConv2D(64, (5, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(DepthwiseConv2D((1, 5), strides=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
