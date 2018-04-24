"""
main file for fitting ARIMA models to visits sereis for each wikipedia page
"""

import sys
import argparse
import numpy as np
import pandas as pd
from . import arima_model

def info():
    """
    System information
    """
    print('Python version: ', sys.version)
    print('Numpy version: ', np.__version__)
    print('Pandas version: ', pd.__version__)

def main():
    parser = argparse.ArgumentParser(
        description = 'Web Traffic Forecasting',
        argument_default = argparse.SUPPRESS
    )
    options = parser.add_subparsers()

    # print information
    op = options.add_parser('info', description = 'Print system information')
    op.set_defaults(func = info)

    # Required args
    parser.add_argument("-d", "--data_path", default = 'data/train_1.csv',
        help = "Path to the training series ")
    parser.add_argument("-k", "--keys_path", default = 'data/key_1.csv',
        help = "Path to the key list of pages")
    # Optional args
    parser.add_argument("-n", "--pred_days", type = int, default = 60,
        help = "Number of days to forecasts, [Default: 60]")
    parser.set_defaults(func = arima_model.main)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
