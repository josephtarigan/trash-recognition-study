import argparse
from trainer import TrashRecognitionInceptionV3Custom1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training-set-folder',
        nargs   = '+',
        help    = 'Training set folder',
        default = 'D:/Workspaces/python/TrashRecognition/trainer/dataset'
    )
    parser.add_argument(
        '--validation-set-folder',
        nargs='+',
        help='Validation set folder',
        default='D:/Workspaces/python/TrashRecognition/trainer/validation_dataset'
    )
    parser.add_argument(
        '--output-dir',
        nargs   = '+',
        help    = 'Output folder',
        default = 'D:/Workspaces/python/output/'
    )
    parser.add_argument(
        '--step-per-epoch',
        type    = int,
        help    = 'Step per epoch',
        default = 2527
    )
    parser.add_argument(
        '--epoch',
        type    = int,
        help    = 'Number of training epoch',
        default = 200
    )
    parser.add_argument(
        '--batch-size',
        type    = int,
        help    = 'Training batch size',
        default = 48
    )
    parser.add_argument(
        '--lr',
        type    = float,
        help    = 'Learning rate',
        default = '0.045'
    )

    args, _ = parser.parse_known_args()
    print(args)
    TrashRecognitionInceptionV3Custom1.do_training(args.training_set_folder[0], args.output_dir[0], args.step_per_epoch, args.epoch, args.batch_size, args.lr)

