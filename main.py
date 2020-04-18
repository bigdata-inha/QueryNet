"""
__author__ = "Jin YeongHwa 진영화"
train
-Capture the configs file
-Process the json configs passed
-Create an agent instance
-Run the agent
"""

import argparse
import easydict

from utils.config import *
from agents import *

# shell command: python main.py configs/vgg_exp_imagenet_0.json

def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()


if __name__ == '__main__':
    main()
