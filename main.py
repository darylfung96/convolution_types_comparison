from keras.callbacks import TensorBoard
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from models.depthwise_conv import DepthWiseConv
from models.standard_conv import StandardConv
from models.standard_eff_conv import StandardEffConv
from models.depthwise_eff_conv import DepthWiseEffConv

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

X_train /= 255
X_test /= 255

# standard_model = StandardConv(input_shape=(28, 28, 1))
# standard_model.summary()
# standard_model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# standard_eff_model = StandardEffConv(input_shape=(28, 28, 1))
# standard_eff_model.summary()
# standard_eff_model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
#
# depthwise_model = DepthWiseConv(input_shape=(28, 28, 1))
# depthwise_model.summary()
# depthwise_model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
#
depthwise_eff_model = DepthWiseEffConv(input_shape=(28, 28, 1))
depthwise_eff_model.summary()
depthwise_eff_model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

