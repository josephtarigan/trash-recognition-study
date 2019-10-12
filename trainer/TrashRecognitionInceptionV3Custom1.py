'''
Inception V3 Transfer Learning Model
joseph.tarigan@gmail.com
'''
import os

import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from keras.models import Model
from tensorflow.python.lib.io import file_io

from trainer.util import DatasetUtil
from trainer.util import ModelCheckpointGc

## Global Variables
CLASSES = 6

'''
inception v3 base model
'''


def create_base_model():
    input_tensor = Input(shape=(244, 244, 3))
    return InceptionV3(input_tensor=input_tensor, include_top=False, weights='imagenet')


'''
custom discriminative layer
'''


def create_custom_model():
    base_model = create_base_model()
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    discriminative_layer = Dense(CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=discriminative_layer)

    for layer in base_model.layers:
        layer.trainable = False

    return model


'''
history util
'''


def show_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.show()


def save_to_gcp(remote_path, file_path):
    with file_io.FileIO(file_path, 'rb') as input_f:
        with file_io.FileIO(remote_path, 'wb') as output_f:
            output_f.write(input_f.read())


def do_training(training_folder_path, output_model_dir, steps_per_epoch, epoch, batch_size, learning_rate):
    # tensorboard callback init
    tensorboard = TensorBoard(
        os.path.join(output_model_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0,
        update_freq='epoch'
    )

    # save model at every epoch
    checkpoint = ModelCheckpoint(os.path.join(output_model_dir, 'checkpoint') + '/checkpoint-{epoch:02d}.mdl', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    if output_model_dir.split(':')[0] == 'gs':
        checkpoint = ModelCheckpointGc.ModelCheckpointGC(checkpoint)

    # callbacks
    if output_model_dir.split(':')[0] == 'gs':
        callbacks = [checkpoint]
    else:
        callbacks = [tensorboard, checkpoint]

    # dataset
    if training_folder_path.split(':')[0] == 'gs':
        train_dataset, label_dataset, total_files = DatasetUtil.load_dataset(training_folder_path)
        # train_dataset = DatasetUtil.load_dataset(training_folder_path, batch_size)
    else:
        #train_dataset = DatasetUtil.load_dataset_flow_from_dir(training_folder_path, batch_size)
        train_dataset, label_dataset, total_files = DatasetUtil.load_local_dataset(training_folder_path)

    # load the model
    inception_model = create_custom_model()
    print(inception_model.summary())

    # train the model
    inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # train the model
    if training_folder_path.split(':')[0] == 'gs':
        history = inception_model.fit(train_dataset
                                      , label_dataset
                                      , batch_size=batch_size
                                      , epochs=epoch
                                      , verbose=2
                                      , callbacks=callbacks)
        # history = inception_model.fit_generator(train_dataset
        #                                         , steps_per_epoch=steps_per_epoch
        #                                         , epochs=epoch
        #                                         , verbose=2
        #                                         , callbacks=callbacks)
    else:
        history = inception_model.fit_generator(train_dataset
                                                , steps_per_epoch=steps_per_epoch
                                                , epochs=epoch
                                                , verbose=2
                                                , callbacks=callbacks)
        # history = inception_model.fit(train_dataset
        #                               , label_dataset
        #                               , validation_split=0.2
        #                               , batch_size=batch_size
        #                               , epochs=epoch
        #                               , verbose=2
        #                               , callbacks=callbacks)

    # save the model
    if training_folder_path.split(':')[0] == 'gs':
        inception_model.save('inception-v3-model.h5')
        save_to_gcp(os.path.join(output_model_dir, 'inception-v3-model.h5'), 'inception-v3-model.h5')
    else:
        inception_model.save(os.path.join(output_model_dir, 'inception-v3-model.h5'))

    # show history
    if training_folder_path.split(':')[0] != 'gs':
        show_history(history)
