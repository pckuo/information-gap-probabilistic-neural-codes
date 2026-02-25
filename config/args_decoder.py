# for submission to ICLR 2026

import argparse
from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # -- ENVIRONMENT --
    parser.add_argument('--seed',  nargs='+', type=int, default=73)
    parser.add_argument('--stimuli_ori_start', type=int, default=-45)
    parser.add_argument('--stimuli_ori_end', type=int, default=45)
    
    # -- LOGGING --
    parser.add_argument('--results_log_dir', default=None, help='directory to save results (None uses ./logs)')
    parser.add_argument('--log_interval', type=int, default=500, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval, one save per n updates')
    parser.add_argument('--save_intermediate_models', type=boolean_argument, default=False, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=500, help='eval interval, one eval per n updates')
    parser.add_argument('--vis_interval', type=int, default=500, help='visualisation interval, one eval per n updates')
    
    # -- POLICY --
    # NNDecoder setup
    parser.add_argument('--input_dim', type=int, default=100,
                        help='input to the decoder, i.e. num of units')
    parser.add_argument('--nn_layers', nargs='+', type=int, default=[300, 200])
    parser.add_argument('--nn_dropout_rates', nargs='+', type=int, default=[0.5, 0.5])
    parser.add_argument('--output_dim', type=int, default=91,
                        help='output of the decoder, i.e. binned orientation')
    parser.add_argument('--net_initialization_method', type=str, default='normc',
                        help='choose: orthogonal, normc')
    
    # training
    parser.add_argument('--decoder_type', type=str, default='lh', 
                        help='choose: lh, post, flex')
    parser.add_argument('--learning_rate', type=float, default=0.0003, 
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--loss_smoothness_coeff', type=float, default=100.0, 
                        help='loss coefficient for smoothness')
    parser.add_argument('--num_epochs', type=int, default=5e2,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--early_stopping', type=boolean_argument, default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='patience for early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4,
                        help='minimum change to qualify as an improvement')
    parser.add_argument('--early_stopping_restore_best', type=boolean_argument, default=True,
                        help='restore best model weights when stopping')

    return parser.parse_args(rest_args)