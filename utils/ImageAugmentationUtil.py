from keras.preprocessing.image import ImageDataGenerator
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        nargs='+',
        help='Input folder'
    )
    parser.add_argument(
        '--output-dir',
        nargs='+',
        help='Output folder'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        help='Image target size',
        default=224
    )

    args, _ = parser.parse_known_args()

    dataset_generator = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=0.3,
        width_shift_range=0.3
    )

    iterator = dataset_generator.flow_from_directory(directory=args.input_dir[0], target_size=(args.target_size, args.target_size), class_mode='categorical', save_to_dir=args.output_dir[0])

    i = 100
    for x, y in iterator:
        print('{0}'.format(y))
        i=i+1
        if i == 100:
            exit(0)