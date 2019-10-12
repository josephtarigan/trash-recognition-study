from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils.np_utils import to_categorical
from google.cloud import storage
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
import os
from io import BytesIO

labels = {'cardboard': 0,
          'glass': 1,
          'metal': 2,
          'paper': 3,
          'plastic': 4,
          'trash': 5}


def get_labels():
    return labels


def load_local_dataset(path):
    x = list()
    y = list()

    for _path, directories, files in os.walk(path):
        for file in files:
            i = load_img(_path + '/' + file)
            a = img_to_array(i)
            class_names = _path.split('/')
            class_name = class_names[len(class_names) - 1]

            x.append(a)
            y.append(get_labels().get(class_name))

    y = to_categorical(y, len(get_labels()))
    return np.array(x), y, 0


'''
load dataset util
will check whether the passed path is google cloud or it doesn't
if the path is google cloud path, TODO
if the path is a local path, then it will directly load the data from given path
'''


def load_dataset(path):
    x = list()
    y = list()
    client = storage.Client()
    bucket = client.get_bucket(path.split('//')[1].split('/')[0])
    blobs = list(bucket.list_blobs(prefix=path.split('//')[1].split('/')[1]))
    total_files = len(blobs)

    for blob in blobs:
        dirs = blob.name.split('/')
        if blob.content_type != 'image/jpeg':
            continue
        else:
            class_name = dirs[1]
            gs_name = '{0}/{1}'.format('gs://' + path.split('//')[1].split('/')[0], blob.name)
            f = BytesIO()
            blob.download_to_file(f)
            i_f = load_img(f)
            a = img_to_array(i_f)

            x.append(a)
            y.append(get_labels().get(class_name))

    print('{0}{1}{2}'.format('loaded ', len(x), ' images'))

    xx = np.array(x)
    yy = to_categorical(y, len(get_labels()))
    x = None
    y = None

    return xx, yy, total_files

    # gcs_dir = './{0}/'.format(path.split('//')[1].split('/')[1])
    # print('GCS dir is {0}'.format(gcs_dir))

    # train_generator = datagen.flow_from_directory(
    #     directory=gcs_dir,
    #     target_size=(244, 244),
    #     class_mode='categorical',
    #     shuffle=True,
    #     batch_size=batch_size
    # )
    #
    # return train_generator


def load_dataset_flow_from_dir(path, batch_size):
    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_directory(
        directory=path,
        target_size=(244, 244),
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size
    )

    return train_generator


def load_random_data_as_validation(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        width_shift_range=200.0,
        height_shift_range=200.0
    )
    train_generator = datagen.flow_from_directory(
        directory=path,
        target_size=(244, 244),
        class_mode='categorical',
        shuffle=False,
        seed=42,
        batch_size=24
    )
    return train_generator