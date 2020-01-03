import configargparse as cfargparse


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(self,
                 config_file_parser_class=cfargparse.YAMLConfigFileParser,
                 formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
                 **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class, **kwargs)
