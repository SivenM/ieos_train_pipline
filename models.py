import tensorflow as tf
from tensorflow import keras


class SeparableConv(keras.layers.Layer):

    def __init__(self, pointwise_conv_filters, strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.dw_layer = keras.layers.DepthwiseConv2D((3, 3),
                               padding='same',
                               depth_multiplier=1,
                               strides=strides,
                               use_bias=False)
        self.batchnorm_1 = keras.layers.BatchNormalization()
        self.batchnorm_2 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.ReLU()
        self.relu_2 = keras.layers.ReLU()
        self.pointwise_conv = keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                                                padding='same',
                                                use_bias=False,
                                                strides=(1, 1),
                                                kernel_regularizer=keras.regularizers.l2(0.0001)
                                                )
    
    def call(self, X):
        x = self.dw_layer(X)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)
        x = self.pointwise_conv(x)
        x = self.batchnorm_2(x)
        x = self.relu_2(x)
        return x


class Model32(keras.Model):

    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchor_boxes = 1
        self.output_dim = self.num_anchor_boxes * (self.num_classes + 4)
        self.input_conv = keras.layers.Conv2D(32, (3,3), 
                                            padding='same', activation='relu', name='input_conv')

        self.sep_32 = SeparableConv(64) 
        self.sep_16 = SeparableConv(128, (2, 2)) 
        self.sep_8 = SeparableConv(256, (2, 2)) 
        self.sep_4 = SeparableConv(256, (2, 2)) 
        self.sep_2 = SeparableConv(512, (2, 2)) 
        self.sep_1 = SeparableConv(1024, (2, 2)) 
        self.dropout_layer = keras.layers.Dropout(rate=0.5)
        self.out_conv = keras.layers.Conv2D(filters=self.output_dim, kernel_size=(1, 1), padding='same')

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_classes": self.num_classes,
                "num_anchor_boxes": self.num_anchor_boxes,
                "output_dim": self.output_dim,
                "input_conv": self.input_conv,
                "sep_32": self.sep_32,
                "sep_32": self.sep_16,
                "sep_32": self.sep_8,
                "sep_32": self.sep_4,
                "sep_32": self.sep_2,
                "sep_32": self.sep_1,
                "dropout_layer": self.dropout_layer,
                "out_conv": self.out_conv
                }

    def call(self, img_array, training=None):
        input_layer = self.input_conv(img_array)
        x = self.dropout_layer(input_layer, training=training)
        x = self.sep_32(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_16(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_8(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_4(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_2(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_1(x)
        return self.out_conv(x)



class Model64(keras.Model):

    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchor_boxes = 1
        self.output_dim = self.num_anchor_boxes * (self.num_classes + 4)
        self.input_conv = keras.layers.Conv2D(32, (3,3), 
                                            padding='same', activation='relu', name='input_conv')

        self.sep_32 = SeparableConv(64, (2, 2)) 
        self.sep_16 = SeparableConv(128, (2, 2)) 
        self.sep_8 = SeparableConv(256, (2, 2)) 
        self.sep_4 = SeparableConv(256, (2, 2)) 
        self.sep_2 = SeparableConv(512, (2, 2)) 
        self.sep_1 = SeparableConv(1024, (2, 2)) 
        self.dropout_layer = keras.layers.Dropout(rate=0.5)
        self.out_conv = keras.layers.Conv2D(filters=self.output_dim, kernel_size=(1, 1), padding='same')
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_classes": self.num_classes,
                "num_anchor_boxes": self.num_anchor_boxes,
                "output_dim": self.output_dim,
                "input_conv": self.input_conv,
                "sep_32": self.sep_32,
                "sep_32": self.sep_16,
                "sep_32": self.sep_8,
                "sep_32": self.sep_4,
                "sep_32": self.sep_2,
                "sep_32": self.sep_1,
                "dropout_layer": self.dropout_layer,
                "out_conv": self.out_conv
                }

    def call(self, img_array, training=None):
        input_layer = self.input_conv(img_array)
        x = self.dropout_layer(input_layer, training=training)
        x = self.sep_32(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_16(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_8(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_4(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_2(x)
        x = self.dropout_layer(x, training=training)
        x = self.sep_1(x)
        return self.out_conv(x)




def separable_conv(pointwise_conv_filters, strides=(1, 1)):
    return keras.Sequential(
    [
        keras.layers.DepthwiseConv2D((3, 3),
                                 padding='same',
                                 depth_multiplier=1,
                                 strides=strides,
                                 use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(6.),  
        keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                        padding='same',
                        use_bias=False,
                        strides=(1, 1),
                        ),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(6.)
    ])


def model_32(num_classes=2, training=None):
    num_classes = num_classes
    num_anchor_boxes = 1
    output_dim = num_anchor_boxes * (num_classes + 4)
    input_layer = keras.Input(shape=(32, 32, 1), name='input')
    input_conv = keras.layers.Conv2D(32, (3,3), 
                                            padding='same', activation='relu', name='input_conv')(input_layer)
    x = keras.layers.Dropout(0.5)(input_conv)
    x = separable_conv(64)(x)
    x = keras.layers.Dropout(0.5)(input_conv)
    x = separable_conv(128, (2, 2))(x)
    x = separable_conv(256, (2, 2))(x)
    x = separable_conv(256, (2, 2))(x)
    x = separable_conv(512, (2, 2))(x)
    x = separable_conv(1024, (2, 2))(x)
    x = keras.layers.Conv2D(filters=output_dim, kernel_size=(1, 1), padding='same')(x)
    output = tf.keras.layers.Reshape((1, 6), input_shape=(1, 1, 6))(x)
    return keras.Model(inputs=input_layer, outputs = output)


def model_64(num_classes=2, training=None):
    num_classes = num_classes
    num_anchor_boxes = 1
    output_dim = num_anchor_boxes * (num_classes + 4)
    input_layer = keras.Input(shape=(32, 32, 1), name='input')
    input_conv = keras.layers.Conv2D(32, (3,3), 
                                            padding='same', activation='relu', name='input_conv')(input_layer)
    x = keras.layers.Dropout(0.5)(input_conv)
    x = separable_conv(64, (2, 2))(x)
    x = keras.layers.Dropout(0.5)(input_conv)
    x = separable_conv(128, (2, 2))(x)
    x = separable_conv(256, (2, 2))(x)
    x = separable_conv(256, (2, 2))(x)
    x = separable_conv(512, (2, 2))(x)
    x = separable_conv(1024, (2, 2))(x)
    x = keras.layers.Conv2D(filters=output_dim, kernel_size=(1, 1), padding='same')(x)
    output = tf.keras.layers.Reshape((1, 6), input_shape=(1, 1, 6))(x)
    return keras.Model(inputs=input_layer, outputs = output)
    