import tensorflow as tf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        help='Input model'
    )
    parser.add_argument(
        '--output',
        nargs='+',
        help='Output model'
    )

    args, _ = parser.parse_known_args()

    converter = tf.lite.TFLiteConverter.from_keras_model_file(args.input[0])
    tflite_model = converter.convert()
    open(args.output[0], "wb+").write(tflite_model)