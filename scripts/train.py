import random
import sys

sys.path =  ['.'] + sys.path

import torch
from arguments import training_arguments
from runners.training_runners import training_runners
from utils.common_utils import printer, setup_seed


if __name__ == "__main__":
    config = training_arguments.load_config()
    setup_seed(config.exp.seed)

    printer(config)

    trainer = training_runners[config.train.train_runner](config)
    trainer.setup()
    trainer.run()
