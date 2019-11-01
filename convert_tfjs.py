import tensorflowjs as tfjs
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Activation, Concatenate, GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Permute, multiply, Average
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback
from tensorflow.keras.utils import to_categorical

def squeeze_excite_block(tensor, ratio=16):
    # From: https://github.com/titu1994/keras-squeeze-excite-network
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = K.int_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def create_model():
    pretrained = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling=None)
    x = pretrained.output

    # Original branch
    gavg = GlobalAveragePooling2D()(x)
    gmax = GlobalMaxPooling2D()(x)
    original_concat = Concatenate(axis=-1)([gavg, gmax,])
    original_concat = Dropout(0.5)(original_concat)
    original_final = Dense(133, activation='softmax', name='original_out')(original_concat)

    # SE branch
    se_out = squeeze_excite_block(x)
    se_gavg = GlobalAveragePooling2D()(se_out)
    se_gmax = GlobalMaxPooling2D()(se_out)
    se_concat = Concatenate(axis=-1)([se_gavg, se_gmax,])
    se_concat = Dropout(0.5)(se_concat)
    se_final = Dense(133, activation='softmax', name='se_out')(se_concat)

    combined_output = Average()([original_final, se_final])
    model = Model(inputs=pretrained.input, outputs=combined_output)
    return model

def main():
    model = create_model()
    print("model created")
    model.load_weights('saved_models/weights.best.MobileNetV2_whole_model.hdf5')
    model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
    tfjs.converters.save_keras_model(model, 'tfjs_model')
    print("model converted")
if __name__ == '__main__':
    main()