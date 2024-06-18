import os

from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from utils.class_registry import ClassRegistry
from models.methods import methods_registry
from metrics.metrics import metrics_registry



args = ClassRegistry()


@args.add_to_registry("exp")
@dataclass
class ExperimentArgs:
    config_dir: str = str(Path(__file__).resolve().parent / "configs")
    config: str = MISSING
    output_dir: str = "results_dir"
    seed: int = 1
    root: str = os.getenv("EXP_ROOT", ".")
    domain: str = "human_faces"
    wandb: bool = False


@args.add_to_registry("data")
@dataclass
class DataArgs:
    inference_dir: str = ""
    transform: str = "face_1024"


@args.add_to_registry("inference")
@dataclass
class InferenceArgs:
    inference_runner: str = "base_inference_runner"
    editings_data: Dict = field(default_factory=lambda: {})


@args.add_to_registry("model")
@dataclass
class ModelArgs:
    method: str = "fse_full"
    device: str = "0"
    batch_size: int = 4
    workers: int = 4
    checkpoint_path: str = ""



MethodsArgs = methods_registry.make_dataclass_from_args("MethodsArgs")
args.add_to_registry("methods_args")(MethodsArgs)

MetricsArgs = metrics_registry.make_dataclass_from_args("MetricsArgs")
args.add_to_registry("metrics")(MetricsArgs)



Args = args.make_dataclass_from_classes("Args")


def load_config():
    config = OmegaConf.structured(Args)

    conf_cli = OmegaConf.from_cli()
    config.exp.config = conf_cli.exp.config
    config.exp.config_dir = conf_cli.exp.config_dir

    config_path = os.path.join(config.exp.config_dir, config.exp.config)
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, conf_file)
    for method in list(config.methods_args.keys()):
        if method != config.model.method:
            config.methods_args.__delattr__(method)

    config = OmegaConf.merge(config, conf_cli)

    return config
