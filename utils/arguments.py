import argparse


def get_args():

    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_data_config_args(parser)
    parser = add_transform_config_args(parser)
    parser = add_optimizer_config_args(parser)
    parser = add_misc_config_args(parser)

    args = parser.parse_args()

    args_dict = vars(args)

    # Change all arguments that are lists into either single values or tuples.
    for k, v in args_dict.items():
        if isinstance(v, list):
            if len(v) == 1:
                setattr(args, k, v[0])
            else:
                setattr(args, k, tuple(v))

    return args


def add_model_config_args(parser):
    group = parser.add_argument_group('model', 'Model type and settings')

    group.add_argument('--model', default='UNETR_Official', type=str,
                        help='Model name')

    group.add_argument('--vol_size', nargs='*', default=[224], type=int,
                        help='volume input size, can be a single number or for example --input_size 128 64 32 as H*W*D')
    group.add_argument('--patch_size', nargs='*', default=[16], type=int,
                        help='patch input size, can be a single number or for example --patch_size 128 64 32 as H*W*D')
    group.add_argument('--input_dim', default=3, type=int,
                        help='Dimension of the input, allowed values are 2 and 3')
    group.add_argument('--output_dim', default=3, type=int,
                        help='Dimension of the output, must be equal to the number of classes being segmented.')
    group.add_argument('--in_chans', default=1, type=int,
                        help='Number of channels for the volumes')

    group.add_argument('--rel_pos_bias', action='store_true',
                        help='Use relative position bias in the encoder')
    group.set_defaults(rel_pos_bias=False)

    group.add_argument('--abs_pos_emb', action='store_true',
                        help='Use absolute position emb in the encoder')
    group.set_defaults(abs_pos_bias=False)

    group.add_argument('--qkv_bias', action='store_true',
                        help='Use bias for attention qkv in the encoder')
    group.set_defaults(qkv_bias=False)

    group.add_argument('--gradient_clipping', type=float,
                       help='Sets the gradient clipping to the specified number. Gradient clipping disabled when None')
    group.set_defaults(gradient_clipping=None)

    group.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision for model, operations and input')
    group.set_defaults(mixed_precision=False)


    return parser


def add_transform_config_args(parser):
    group = parser.add_argument_group('data', 'Settings for data, both reading from disk and loading into model')

    group.add_argument('--t_voxel_spacings', action='store_true',
                       help='Transform all data into the same voxel spacings')
    group.set_defaults(t_voxel_spacings=False)
    group.add_argument('--t_voxel_dims', nargs='*', default=[1.0], type=float,
                        help='The spacings to transform the voxels into. Can be a single value or a tuple of 3.')

    group.add_argument('--t_cubed_ct_intensity', action='store_true',
                       help='Take the cube root of all ct intensity values to make the input space smaller')
    group.set_defaults(t_cubed_ct_intensity=False)

    group.add_argument('--t_fixed_ct_intensity', action='store_true',
                       help='Clip and normalize ct intensity range to the fixed range between ct min and ct max')
    group.set_defaults(t_fixed_ct_intensity=False)

    group.add_argument('--t_ct_min', default=-1000, type=int,
                        help='The minimum CT intensity value to clip to')
    group.add_argument('--t_ct_max', default=1000, type=int,
                       help='The maximum CT intensity value to clip to')

    group.add_argument('--t_crop_foreground', action='store_true',
                       help='Crop volumes of space that consists of background only')
    group.set_defaults(t_crop_foreground=False)

    group.add_argument('--t_spatial_pad', action='store_true',
                       help='Pad volumes to the input volume size')
    group.set_defaults(t_spatial_pad=False)


    group.add_argument('--t_normalize', action='store_true',
                       help='Pad volumes to the input volume size')
    group.set_defaults(t_normalize=False)
    group.add_argument('--t_norm_mean', default=0.1943, type=float,
                       help='The probability for a random flip in a direction')
    group.add_argument('--t_norm_std', default=0.2786, type=float,
                       help='The probability for a random flip in a direction')


    group.add_argument('--t_n_samples', default=8, type=int,
                       help='The number of sampels for the random cropping')

    group.add_argument('--t_flip_prob', default=0.1, type=float,
                       help='The probability for a random flip in a direction')

    group.add_argument('--t_rot_prob', default=0.1, type=float,
                       help='The probability for a random rotate in a direction')

    group.add_argument('--t_intensity_shift_os', default=0.1, type=float,
                       help='The offset for random intensity shift')
    group.add_argument('--t_intensity_shift_prob', default=0.5, type=float,
                       help='The probability for a random intensity shift')

    return parser

def add_data_config_args(parser):
    group = parser.add_argument_group('transform', 'Settings for transform, both training and validation')

    group.add_argument('--data_path', default='/datasets/', type=str,
                        help='Dataset path')
    group.add_argument('--batch_size_val', type=int, default=1, help='Batch size for validation data loader')
    group.add_argument('--batch_size_train', type=int, default=1, help='Batch size for training data loader')
    group.add_argument('--n_workers_train', type=int, default=8, help='Number of workers used in Train DataLoader')
    group.add_argument('--n_workers_val', type=int, default=4, help='Number of workers used in Val DataLoader')
    group.add_argument('--no_pin_memory', action='store_false', dest='pin_mem',
                       help='When enabled Dataloaders wont pin GPU memory')
    group.set_defaults(pin_mem=True)

    return parser

def add_optimizer_config_args(parser):
    group = parser.add_argument_group('optimizer', 'Optimzer settings')

    group.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    group.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    group.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    return parser

def add_misc_config_args(parser):
    group = parser.add_argument_group('training', 'Training settings')

    group.add_argument('--seed', type=int, default=13, help='Random seed for determinstic training')
    group.add_argument('--no_cuddn_auto_tuner', action='store_true', help='Disables the CUDDN benchmark in PyTorch')
    group.add_argument('--anomaly_detection', action='store_true', help='Enables PyTorch anomaly detection')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    group.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    group.add_argument('--save_ckpt_freq', default=20, type=int)
    group.add_argument('--val_interval', default=20, type=int)
    group.add_argument('--log_dir', type=str, help='Folder where the logs should be saved')
    group.add_argument('--output_dir', type=str, help='path where to save, empty for no saving')
    group.add_argument('--resume', default='', help='resume from checkpoint')
    group.add_argument('--pretrained', type=str, help='Pretrained checkpoint for backbone')

    # distributed training parameters
    group.add_argument('--world_size', default=1, type=int,  help='number of distributed processes')
    group.add_argument('--local_rank', default=-1, type=int)
    group.add_argument('--dist_on_itp', action='store_true')
    group.add_argument('--dist_url', default='env://',  help='url used to set up distributed training')

    return parser