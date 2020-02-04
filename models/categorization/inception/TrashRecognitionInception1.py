'''
Inception inspired model
joseph.tarigan@gmail.com
'''
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('D:\\Workspaces\\python\\trash-recognition\\experiment-models')
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPool2D, Concatenate, Dropout, AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from trainer.dataset import DatasetUtil


def inception_module(input, filter_size):
    
    tower1 = Conv2D(filter_size, (1, 1), activation='relu', padding='same') (input)

    tower2 = Conv2D(filter_size, (1, 1), activation='relu', padding='same') (input)
    tower2 = Conv2D(filter_size, (3, 3), padding='same', activation='relu') (tower2)

    tower3 = Conv2D(filter_size, (1, 1), padding='same', activation='relu') (input)
    tower3 = Conv2D(filter_size, (5, 5), padding='same', activation='relu') (tower3)

    tower4 = MaxPool2D((3, 3), strides=(1, 1), padding='same') (input)
    tower4 = Conv2D(filter_size, (1, 1), padding='same', activation='relu') (tower4)

    concat_layer = Concatenate(axis=3) ([tower1, tower2, tower3, tower4])

    return concat_layer


def create_inception_model ():
    input_layer = Input(shape=(244, 244, 3), name='input')

    conv1 = Conv2D(64, (7, 7), strides=2, activation='relu', name='conv1') (input_layer)
    max_pool1 = MaxPool2D((3, 3), strides=2, padding='same', name='max_pool1') (conv1)

    conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', name='conv2') (max_pool1)
    max_pool2 = MaxPool2D((3, 3), strides=2, padding='same', name='max_pool2') (conv2)

    inception1 = inception_module(max_pool2, 64)

    max_pool3 = MaxPool2D((3, 3), strides=2, padding='same', name='max_pool3') (inception1)

    inception2 = inception_module(max_pool3, 128)
    inception3 = inception_module(inception2, 256)

    max_pool4 = MaxPool2D((3, 3), strides=2, padding='same', name='max_pool4') (inception3)

    inception4 = inception_module(max_pool4, 832)
    inception5 = inception_module(inception4, 1024)

    avg_pool = AveragePooling2D((8, 8), strides=None) (inception5)
    avg_pool = Flatten() (avg_pool)
    avg_pool = Dropout(0.4) (avg_pool)

    ff_layer = Dense(64, activation='relu', name='ff1') (avg_pool)
    output_layer = Dense(6, activation='softmax', name='output') (ff_layer)
    
    inception_model = Model(inputs = input_layer, outputs = output_layer)

    return inception_model

def show_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.show()

# dataset
train_dataset = DatasetUtil.load_dataset('D:\\Workspaces\\python\\trash-recognition\\experiment-models\\dataset\\resulted')

inception_model = create_inception_model()
print (inception_model.summary())
sgd = SGD(lr=0.00045, momentum=0.9, decay=1e-6, nesterov=True)

# train the model
inception_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train the model
history = inception_model.fit_generator(train_dataset, steps_per_epoch=5054, epochs=2000, verbose=2)

# save the model
json_model = inception_model.to_json()
with open("D:\\Workspaces\\python\\trash-recognition\\experiment-models\\inception-model.json", "w+") as json_file:
    json_file.write(json_model)
inception_model.save_weights(os.path.join(os.getcwd(), 'D:\\Workspaces\\python\\trash-recognition\\experiment-models\\inception-model.h5'))

# show history
show_history(history)