import os
import argparse
from keras.models import load_model
from trainer.util import DatasetUtil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        nargs='+',
        help='Validation Dataset',
    )
    parser.add_argument(
        '--model',
        nargs='+',
        help='Model path'
    )

    args, _ = parser.parse_known_args()

    model = load_model(args.model[0])
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    data_generator, class_indices = DatasetUtil.load_random_data_as_validation(args.dataset[0], 32, classes)

    print (class_indices)
    print (model.metrics_names)

    result = model.evaluate_generator(data_generator, steps=data_generator.samples/32, verbose=1, use_multiprocessing=False)

    print(result)