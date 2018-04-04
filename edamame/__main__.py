import argparse

import edamame


def info(args):
    '''Print system info.
    '''
    import sys
    print('Python version:', sys.version)


def main():
    parser = argparse.ArgumentParser(
        description='Web Traffic Prediction',
        argument_default=argparse.SUPPRESS,
    )
    subcommands = parser.add_subparsers()

    # info
    cmd = subcommands.add_parser('info', description='print system info')
    cmd.set_defaults(func=info)

    # Each subcommand gives an `args.func`.
    # Call that function and pass the rest of `args` as kwargs.
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
