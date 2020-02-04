import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        nargs='+',
        help='Model path'
    )
    parser.add_argument(
        '--input',
        nargs='+',
        help='Input image path'
    )

    args, _ = parser.parse_known_args()

    model = load_model(args.model[0])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    img = image.load_img(args.input[0], target_size=(244, 244))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    img = np.vstack([img])

    classes = model.predict(img)

    print(classes)