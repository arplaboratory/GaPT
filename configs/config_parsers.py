import argparse
import sys
import logging
import os
from logging import config
from pathlib import Path
import json
from configs import LOG_CONFIG, MATLAB_CONFIG


class ArgParser:
    """
     The class is used to use the cli argument -configs to load the configs.json file a
     nd parse it. The getter (with decorator @args) returns the parsed arguments.
     Usage:
        - launch the test_train_csv.py with the -c or --configs within the absolute path of the configs file;
        - create a CustomParser object
        - call the getter '.args'
     """

    def __init__(self):
        cli_parser = argparse.ArgumentParser()
        cli_parser.add_argument(
            '-c',
            '--config_file',
            dest='config_file',
            type=str,
            default=None,
            help='configs file',
        )

        self._args, _ = cli_parser.parse_known_args()
        parser = argparse.ArgumentParser(parents=[cli_parser], add_help=False)
        if (self._args.config_file is not None) and ('.json' in self._args.config_file):
            try:
                self._json = json.load(open(self._args.config_file))
                parser.set_defaults(**self._json)
                self._args = parser.parse_args()
            except FileNotFoundError as e:
                logging.error("Unable to open the .json file {}, the returned error is: {}"
                              .format(self._args.config_file, e))
                sys.exit(0)

    @property
    def args(self):
        return self._args

    @property
    def json(self):
        return self._json


class MatplotlibParser:
    @ staticmethod
    def get_matplot_conf():
        try:
            matlab_conf = json.load(open(MATLAB_CONFIG))
            return matlab_conf
        except FileNotFoundError as e:
            logging.error("Unable to open the .json file {}, the returned error is: {}"
                          .format(MATLAB_CONFIG, e))
            sys.exit(0)


class LoggerParser:
    @staticmethod
    def setup_logging(save_dir, default_level=logging.INFO):
        """
        Setup logger
        """
        log_config = Path(LOG_CONFIG)
        save_dir = Path(save_dir)
        try:
            conf = json.load(open(log_config))
            # modify logger paths based on run configs
            for _, handler in conf['LOGGING']['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(os.path.join(save_dir, handler['filename']))
            logging.config.dictConfig(conf['LOGGING'])
        except FileNotFoundError as e:
            print("Unable to open the .json file {}, the returned error is: \n\t->{}".format(log_config, e))
            logging.basicConfig(level=default_level)
        except Exception as e:
            print("An error occur while setting up the logging: \n\t->{}".format(e))
            logging.basicConfig(level=default_level)

