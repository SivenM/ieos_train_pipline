import tensorflow as tf
import numpy as np
from tensorflow import keras
from data_preprocessing import AncorBoxCreator
from callbacks import ModelCheckPoint, History
import utils
import os

class Coach:

    def __init__(self, img_size=64):
        self.img_size = img_size
        self.mean_loss = tf.keras.metrics.Mean()
        self.rmse = tf.keras.metrics.RootMeanSquaredError()
        self.acc = tf.keras.metrics.BinaryAccuracy()
        self.auc = tf.keras.metrics.AUC()

    def update_metrics(self, best_box, best_conf, gt_box, gt_cls):
        self.rmse.update_state(gt_box, best_box)
        self.acc.update_state(gt_cls, best_conf)
        self.auc.update_state(gt_cls, best_conf)

    def reset_metrics(self):
        self.rmse.reset_states()
        self.acc.reset_states()
        self.auc.reset_states()
        self.mean_loss.reset_states()
    
    def result_metrics(self):
        epoch_loss = self.mean_loss.result().numpy()
        epoch_rmse = self.rmse.result().numpy()
        epoch_acc = self.acc.result().numpy()
        epoch_auc = self.auc.result().numpy()
        return epoch_loss, epoch_rmse, epoch_acc, epoch_auc

    def get_boxes_cls(self, y_pred):
        anchor_boxes = AncorBoxCreator(img_size=self.img_size).create_boxes()
        box_predictions = y_pred[:, :, :4]
        cls_predictions = tf.nn.softmax(y_pred[:, :, 4:])
        boxes = utils.decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        return boxes, cls_predictions

    def get_best_box_conf(self, boxes, cls_predictions):
        best_box = []
        best_conf = []
        idxs = tf.argmax(cls_predictions[:,:,0], axis=1).numpy()
        for i, idx in enumerate(idxs):
            best_box.append(boxes[i, idx])
            best_conf.append(cls_predictions[i, idx, 0])
        return np.array(best_box), np.array(best_conf)

    def train_epoch(self, model, train_loss, optimizer, train_dataset, ):
        """
        Эпоха обучения модели. Перед обучением проводится кроссвалидация.
        Оценивается метриками RMSE, Accurasy, AUC
        """

        for i, item in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(item[0], training=True) 
                loss = train_loss(item[1], y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            self.mean_loss(loss)
            gt_cls = item[1][:,0,4]
            gt_box = item[2]
            boxes, cls_predictions = self.get_boxes_cls(y_pred)
            best_box, best_conf = self.get_best_box_conf(boxes, cls_predictions)
            self.update_metrics(best_box, best_conf, gt_box, gt_cls)
        epoch_loss, epoch_rmse, epoch_acc, epoch_auc = self.result_metrics()
        
        self.reset_metrics()       
        return model, epoch_loss, [epoch_rmse, epoch_acc, epoch_auc]

    def val_epoch(self, model, train_loss, val_dataset):
        for i, item in enumerate(val_dataset):
            y_pred = model(item[0], training=True)        
            loss = train_loss(item[1], y_pred)
            self.mean_loss(loss)
            gt_cls = item[1][:,0,4]
            gt_box = item[2]
            boxes, cls_predictions = self.get_boxes_cls(y_pred)
            best_box, best_conf = self.get_best_box_conf(boxes, cls_predictions)
            self.update_metrics(best_box, best_conf, gt_box, gt_cls)
        epoch_loss, epoch_rmse, epoch_acc, epoch_auc = self.result_metrics()        
        self.reset_metrics()       
        return model, epoch_loss, [epoch_rmse, epoch_acc, epoch_auc]

    def parse_cfg(self, cfg):
        return cfg['n_epochs'], cfg['model'], cfg['train_loss'], cfg['optimiser'], cfg['main_dir']

    def fit(self, cfg, train_dataset, val_dataset):
        n_epochs, model, train_loss, optimizer, main_dir = self.parse_cfg(cfg)
        history = History(main_dir, n_epochs)
        checkpoint = ModelCheckPoint(main_dir)
        print('\nНачинаю обучение')
        for epoch in range(n_epochs):
            model, train_epoch_loss, train_metrics = self.train_epoch(model, train_loss, optimizer, train_dataset)
            model, val_epoch_loss, val_metrics = self.val_epoch(model, train_loss, val_dataset)
            print(f'{epoch + 1}. val_loss: {val_epoch_loss} | val_rmse {val_metrics[0]} | val_auc {val_metrics[-1]}')
            history.write((train_epoch_loss, train_metrics), (val_epoch_loss, val_metrics))
            checkpoint.check2(model, val_epoch_loss, epoch)
        history.save_to_json()
        checkpoint.save_data()
        model_name = 'm64_ep_{}.h5'.format(n_epochs)
        save_dir = os.path.join(main_dir, 'models')
        model.save(os.path.join(save_dir, model_name))
        print(f'Обучение на {n_epochs} эпохах завершено.\n')

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
    input_layer = keras.Input(shape=(64, 64, 1), name='input')
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
    x = separable_conv(1024, (2, 2))(x)
    x = keras.layers.Conv2D(filters=output_dim, kernel_size=(1, 1), padding='same')(x)
    output = tf.keras.layers.Reshape((1, 6), input_shape=(1, 1, 6))(x)
    return keras.Model(inputs=input_layer, outputs = output)
    

def model_64_2_outputs(num_classes=2, training=None):
    num_classes = num_classes
    num_anchor_boxes = 1
    output_dim = num_anchor_boxes * (num_classes + 4)
    input_layer = keras.Input(shape=(64, 64, 1), name='input')
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
    x = separable_conv(1024, (2, 2))(x)
    
    box_prediction = keras.layers.Conv2D(4, kernel_size=(1, 1), padding='same')(x)
    box_prediction = tf.keras.layers.Reshape((4), input_shape=(1, 1, 4), name='box')(box_prediction)
    conf_predicion = keras.layers.Conv2D(2, kernel_size=(1, 1), activation='softmax', padding='same')(x)
    conf_predicion = tf.keras.layers.Reshape((2), input_shape=(1, 1, 2), name='conf')(conf_predicion)
    return keras.Model(inputs=input_layer, outputs = [box_prediction, conf_predicion])