import sys
import torch

sys.path =  ['.'] + sys.path

from arguments import inference_arguments
from runners.inference_runners import inference_runner_registry
from utils.common_utils import printer, setup_seed


def run_inference(config):
    inference_runner = inference_runner_registry[config.inference.inference_runner](
        config
    )
    inference_runner.setup()
    inference_runner.run()


if __name__ == "__main__":
    config = inference_arguments.load_config()
    setup_seed(config.exp.seed)

    printer(config)

    run_inference(config)
