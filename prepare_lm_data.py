import albert.configs as configs
from albert.bin import prepare
from albert.utils import ArgumentParser


def _get_parser():
    parser = ArgumentParser(description='prepare.py')
    configs.prepare_data_opts(parser)
    return parser


def main():
    parser = _get_parser()
    conf = parser.parse_args()
    prepare(conf)


if __name__ == "__main__":
    main()
