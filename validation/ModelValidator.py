import numpy as np
import pandas as pd
import argparse
from keras.models import model_from_json
from trainer.util import DatasetUtil

'''
Model validation utility
Output: Filename, Preticted class
In csv format
'''
def validate(model_path, dataset_path, steps, output_path):
    with open(model_path, 'r') as json_model:
        inception_model = model_from_json(json_model.read())
        
        # model summary
        inception_model.summary()

        # load data
        test_dataset = DatasetUtil.load_random_data_as_validation(dataset_path)
        test_dataset.reset()

        # test
        result = inception_model.predict_generator(test_dataset, steps=steps, verbose=1)
        resulted_class_indices = np.argmax(result, axis=1)

        # map the result
        labels = test_dataset.class_indices
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in resulted_class_indices]

        # result
        print (predictions)

        filenames=test_dataset.filenames
        print (len(filenames))
        print (len(predictions))
        results=pd.DataFrame({"Filename":filenames,
                            "Predictions":predictions})
        results.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        nargs='+',
                        help='Model path')

    parser.add_argument('--dataset-path',
                        nargs='+',
                        help='Dataset Path')

    parser.add_argument('--steps',
                        type=int,
                        help='Step per Epoch (Predict Generator)')

    parser.add_argument('--output-path',
                        nargs='+',
                        help='Output Path')

    args, _ = parser.parse_known_args()

    validate(args.model_path[0], args.dataset_path[0], args.steps, args.output_path[0])