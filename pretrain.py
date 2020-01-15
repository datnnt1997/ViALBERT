import albert.configs as configs
from albert.bin.run_pretrain import train
from albert.utils import ArgumentParser


def _get_parser():
    parser = ArgumentParser(description='train.py')
    configs.model_opts(parser)
    configs.pretrain_opts(parser)
    return parser


def main():
    parser = _get_parser()
    conf = parser.parse_args()
    train(conf)


if __name__ == "__main__":
    main()
