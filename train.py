import albert.configs as configs
import numpy as np

from albert.utils import ArgumentParser
from albert.models.model import AlbertModel

def train(opt):
    albert = AlbertModel(opt)
    print(albert)
    params = sum([np.prod(p.size()) for p in albert.parameters()])
    print(params)

def _get_parser():
    parser = ArgumentParser(description='train.py')
    configs.model_opts(parser)

    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()
