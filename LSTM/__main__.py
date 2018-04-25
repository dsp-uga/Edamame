import sys
import argparse
import numpy as np
import pandas as pd
import LSTM.LSTM as lstm

def info():
    """
        System information
        """
    print('Python version: ', sys.version)
    print('Numpy version: ', np.__version__)
    print('Pandas version: ', pd.__version__)

def main():
    parser = argparse.ArgumentParser(description = 'Web Traffic Forecasting',
                                     argument_default = argparse.SUPPRESS)
    options = parser.add_subparsers()

    # print information
    op = options.add_parser('info', description = 'Print system information')
    op.set_defaults(func = info)

    # Required args
    parser.add_argument("-d1", "--train_path", default = 'data/train_1.csv',
                        help = "Path to the training series")
    parser.add_argument("-d2", "--test_path", default = 'data/train_2.csv',
                        help = "Path to the testing series")
    # Optional args
    parser.add_argument("-n", "--pred_days", type = int, default = 60,
                        help = "Number of days to forecasts, [Default: 60]")
    parser.set_defaults(func = lstm.main)
                                     
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
