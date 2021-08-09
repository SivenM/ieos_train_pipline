import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt
import os
import json
import data_preprocessing
import models
import losess
import utils
import tempfile
import argparse


def model_64_2_outputs_optim_rect(num_classes=2):
    
    input_layer = keras.Input(shape=(64, 32, 1), name='input')
    input_conv = keras.layers.Conv2D(4, (3,3), 
                                            padding='same', activation='relu', name='input_conv')(input_layer)
    input_conv = models.separable_conv(input_conv, 4, (1, 1), 'input_layer')
    x = keras.layers.Dropout(0.5)(input_conv)
    x = models.separable_conv(x, 32, (1, 1), 'sep_1')
    x = keras.layers.Dropout(0.5)(x)
    x = models.separable_conv(x, 16, (2, 2), 'sep_2')
    x = keras.layers.Dropout(0.5)(x)
    x = models.separable_conv(x, 128, (2, 2), 'sep_3')
    x = keras.layers.Dropout(0.5)(x)
    x = models.separable_conv(x, 32, (2, 2), 'sep_4')
    x = keras.layers.Dropout(0.5)(x)
    x = models.separable_conv(x, 256, (2, 2), 'sep_5')
    x = keras.layers.Dropout(0.5)(x)
    x = models.separable_conv(x, 128, (2, 2), 'sep_6')
    x = keras.layers.Dropout(0.5)(x)
    
    #outputs
    
    box_prediction = keras.layers.Conv2D(4, kernel_size=(1, 1), activation='relu', name='box_conv1')(x)
    box_prediction = keras.layers.Flatten(name='box_flatten')(box_prediction)
    box_prediction = keras.layers.Dense(4, activation='relu', name='box_out')(box_prediction)

    conf_predicion = keras.layers.Conv2D(2, kernel_size=(1, 1), activation='relu', padding='same', name='conf_conv1')(x)
    conf_predicion = keras.layers.Flatten(name='conf_flatten')(conf_predicion)
    conf_predicion = keras.layers.Dense(2, activation='softmax', name='conf_out')(conf_predicion)
    return keras.Model(inputs=input_layer, outputs = [box_prediction, conf_predicion]) 


def main(image_path, labels_path, img_size, batch_size, epochs):
    train_gen = data_preprocessing.HumanDataset(image_path, labels_path, batch_size, img_size=img_size)
    model_1 = model_64_2_outputs_optim_rect()
    dirs = ['save_models/rect_new_dataset_64x32/']
    #models = [model_1]

    for i, model in zip(dirs,[model_1]):
        model_path = i
        model_save_name = 'model_epoch-{epoch:02d}.h5'
        csv_name = 'model_training_log.csv'
        chpkt_path = model_path + model_save_name
        csv_logger_path = model_path + csv_name
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=chpkt_path,
                                    monitor='loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    )

        csv_logger = tf.keras.callbacks.CSVLogger(filename=csv_logger_path,
                        separator=',',
                        append=True)


        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                            factor=0.2,
                                            patience=10,
                                            verbose=1,
                                            cooldown=0,
                                            min_lr=0.0001)


        callbacks = [model_checkpoint,
                csv_logger,
                reduce_learning_rate]

        #model.compile(optimizer='adam',
        #          loss={
        #              'box_out': 'mse',
        #              'conf_out': 'categorical_crossentropy'
        #          },
        #          loss_weights={'box_out': 0.25,
        #                        'conf_out': 1.})
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=[tf.keras.losses.Huber(), 'categorical_crossentropy'],
                loss_weights=[0.1, 1.]
                )

        model.fit(train_gen, 
            epochs=epochs,
            #validation_data=(img_val, [gt_box_val, gt_conf_val]), validation_batch_size=16,
            callbacks=callbacks,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', dest='config', type=argparse.FileType('r'), default=None, help='cfg file in json format')
    args = parser.parse_args()
    if args.config:
        config = json.load(args.config)
    image_path = os.path.join(config['dataset_path'], 'train_images')
    labels_path = os.path.join(config['dataset_path'], 'train_labels.csv')
    img_size = tuple(config['img_size'])
    batch_size = config['batch_size']
    epochs = config['epochs']
    main(image_path, 
        labels_path,  
        img_size,
        batch_size,
        epochs)