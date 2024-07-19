import os
import sys
import torch
import json
import omegaconf
import wandb
import glob

from pathlib import Path
from editings.latent_editor import LatentEditor

from models.methods import methods_registry
from metrics.metrics import metrics_registry
from utils.model_utils import get_stylespace_from_w


class BaseRunner:
    def __init__(self, config):
        self.config = config
        self.method_config = config.methods_args[config.model.method]

    def setup(self):
        self._setup_device()
        self._setup_latent_editor()
        self._setup_method()

    def get_edited_latent(self, original_latent, editing_name, editing_degrees, original_image=None):
        if editing_name in self.latent_editor.stylespace_directions:
            stylespace_latent = get_stylespace_from_w(original_latent, self.method.decoder)
            edited_latents = (
                self.latent_editor.get_stylespace_edits(
                    stylespace_latent, editing_degrees, editing_name
                ))
        elif editing_name in self.latent_editor.interfacegan_directions:
            edited_latents = (
                self.latent_editor.get_interface_gan_edits(
                    original_latent, editing_degrees, editing_name
                ))

        elif editing_name in self.latent_editor.styleclip_directions:
            edited_latents = self.latent_editor.get_styleclip_mapper_edits(
                original_latent, editing_degrees, editing_name
            )

        elif editing_name in self.latent_editor.ganspace_directions:
            edited_latents = (
                self.latent_editor.get_ganspace_edits(
                    original_latent, editing_degrees, editing_name
                )
            )
        elif editing_name in self.latent_editor.fs_directions.keys():
            edited_latents = self.latent_editor.get_fs_edits(
                    original_latent, editing_degrees, editing_name
                )
        elif editing_name.startswith("styleclip_global_"):
            stylespace_latent = get_stylespace_from_w(original_latent, self.method.decoder)
            edited_latents = (
                self.latent_editor.get_styleclip_global_edits(
                    stylespace_latent, editing_degrees, editing_name.replace("styleclip_global_", "")
                ))
        elif editing_name.startswith("deltaedit_"):
            assert original_image is not None
            stylespace_latent = get_stylespace_from_w(original_latent, self.method.decoder)
            edited_latents = (
                self.latent_editor.get_deltaedit_edits(
                    stylespace_latent, editing_degrees, editing_name.replace("deltaedit_", ""), original_image
                ))
        else:
            raise ValueError(f'Edit name {editing_name} is not available')
        return edited_latents

    def _setup_latent_editor(self):
        self.latent_editor = LatentEditor(self.config.exp.domain)

    def _setup_device(self):
        config_device = self.config.model["device"].lower()

        if config_device == "cpu":
            device = "cpu"
        elif config_device.isdigit():
            device = "cuda:{}".format(config_device)
        elif config_device.startswith("cuda:"):
            device = config_device
        else:
            raise ValueError("Incorrect Device Type")

        try:
            torch.randn(1).to(device)
            print("Device: {}".format(device))
        except Exception as e:
            print("Could not use device {}, {}".format(device, e))
            print("Set device to CPU")
            device = "cpu"

        self.device = torch.device(device)

    def _setup_method(self):
        method_name = self.config.model.method
        self.method = methods_registry[method_name](
            checkpoint_path=self.config.model.checkpoint_path,
            **self.config.methods_args[method_name],
        ).to(self.device)
