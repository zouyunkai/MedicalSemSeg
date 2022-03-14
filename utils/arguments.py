import argparse


def get_args():

    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_data_config_args(parser)
    parser = add_training_config_args(parser)

    args = parser.parse_args()

    return args


def add_model_config_args(parser):
    group = parser.add_argument_group('model', 'Model type and settings')

    group.add_argument('--model_name', type=str, default='resnext', help='Used for model selection and logging',
                       choices=['inception', 'resnext', 'resnet'])
    group.add_argument('--n_outputs', type=int, default=25, help='The number of outputs from the model.')

    group.add_argument('--finetune', default=False, action='store_true', help='Finetuning enabled')
    group.add_argument('--pre_trained_checkpoint', type=str, help='checkpoint used to initialize finetuning')

    return parser


def add_data_config_args(parser):
    group = parser.add_argument_group('data', 'Settings for data, both reading from disk and loading into model')

    group.add_argument('--allowed_classes', nargs='+', type=int, help='Identifiers to allowed view labels.'
                                                                                   'Example: --allowed_views 0 2 4')
    group.add_argument('--source_dataset_folder', type=str, required=True, help='Path to folder containing image data')

    group.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for validation data loader')
    group.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training data loader')
    group.add_argument('--n_workers', type=int, default=11, help='Number of workers used in DataLoader')
    group.add_argument('--keep_last', default=False, action='store_true', help='When enabled, dont drop the last batch '
                                                                               'from DataLoader when batch is not '
                                                                               'full-sized')
    group.add_argument('--no_pin_memory', default=False, action='store_true', help='When enabled Dataloaders wont pin GPU memory')

    return parser


def add_training_config_args(parser):
    group = parser.add_argument_group('training', 'Training settings')

    group.add_argument('--local_rank', type=int, default=-1, help='Used for distributed training')
    group.add_argument('--no_cuddn_auto_tuner', default=False, action='store_true', help='Disables the CUDDN benchmark in PyTorch')
    group.add_argument('--anomaly_detection', default=False, action='store_true', help='Enables PyTorch anomaly detection')
    group.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    group.add_argument('--checkpointing_disabled', action='store_true', help='Disables automatic checkpointing during training')
    group.add_argument('--checkpoint_save_path', type=str,
                       default='/home/ola/Projects/View-Classification/saved_models/',
                       help='Folder where the checkpoint should be saved')
    group.add_argument('--freeze_lower', default=False, action='store_true', help='When enabled, all layers except last are frozen')

    return parser