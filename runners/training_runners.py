import os
import sys
import json
import wandb
import datetime
import omegaconf

import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image
from time import time
from pathlib import Path
from abc import abstractmethod

from runners.base_runner import BaseRunner
from utils.class_registry import ClassRegistry
from datasets.transforms import transforms_registry
from datasets.datasets import ImageDataset
from datasets.loaders import InfiniteLoader
from training.losses import disc_losses, LossBuilder
from training.optimizers import optimizers
from metrics.metrics import metrics_registry

from training.loggers import Timer, StreamingMeans, TrainigLogger
from utils.common_utils import tensor2im, get_keys
from models.methods import methods_registry

from models.psp.encoders.psp_encoders import ProgressiveStage
from utils.model_utils import toogle_grad


training_runners = ClassRegistry()
        

FACE_DIRECTIONS = {
    "age": [-7, -5, 5, 7, 10],
    "fs_makeup": [5, 8, 12],
    "afro": [0.03, 0.07],
    "angry": [0.06, 0.1],
    "purple_hair": [0.07, 0.1, 0.12],
    "glasses": [-10, -7],
    "face_roundness": [-13, -7, 7, 13], 
    "rotation": [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
    "bobcut": [0.07, 0.12, 0.18],
    "bowlcut": [0.07, 0.14],
    "mohawk": [0.07, 0.10],
    "blond hair": [-8, -4, 4, 8],
    "fs_smiling": [-6, -3, 3, 6, 9]
}


def get_random_edit():
    direction = np.random.choice(list(FACE_DIRECTIONS.keys()))
    strenght = np.random.choice(FACE_DIRECTIONS[direction])
    return direction, strenght
        

@training_runners.add_to_registry(name="base_training_runner")
class BaseTrainingRunner(BaseRunner):
    def setup(self):
        self.start_step = self.config.train.start_step
        self._setup_device()
        self._setup_experiment_dir()

        self._setup_method()
        self._setup_logger()

        self._setup_metrics()
        self._setup_datasets()

        start_batch_size = (
            self.config.train.bs_used_before_adv_loss
            if self.config.train.train_dis
            else self.config.model.batch_size
        )

        self._setup_dataloaders(start_batch_size)

        self._setup_latent_editor()
        self._setup_optimizers()
        self._setup_loss()

    def _setup_logger(self):
        self.logger = TrainigLogger(self.config)

    def _setup_datasets(self):
        print("Loading dataset")
        transform_dict = transforms_registry[self.config.data.transform]().get_transforms()
        self.train_dataset = ImageDataset(
            self.config.data.input_train_dir, transform_dict["train"]
        )

        self.test_dataset = ImageDataset(
            self.config.data.input_val_dir, transform_dict["test"]
        )
        self.paths = self.test_dataset.paths

        self.special_dataset = ImageDataset(
            self.config.data.special_dir, transform_dict["test"]
        )
        self.special_paths = self.special_dataset.paths

    def _setup_dataloaders(self, batch_size):
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.model.workers,
            drop_last=True,
            is_infinite=True
        )
        self.test_dataloader = InfiniteLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.model.workers,
            is_infinite=False
        )
        self.special_dataloader = InfiniteLoader(
            self.special_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.model.workers,
            is_infinite=False
        )

    def _setup_optimizers(self):
        params = list(self.method.encoder.parameters())

        optimizer_args = dict(
            self.config.optimizers[self.config.train.encoder_optimizer]
        )
        optimizer_args["params"] = params
        self.encoder_optimizer = optimizers[self.config.train.encoder_optimizer](
            **optimizer_args
        )

        if self.config.model.checkpoint_path != "":
            ckpt = torch.load(self.config.model.checkpoint_path, map_location="cpu")
            if "encoder_opt" in ckpt.keys():
                self.encoder_optimizer.load_state_dict(ckpt["encoder_opt"])
            else:
                print('WARNING, continuing training without loading encoder optimizer state!')

        if self.config.train.train_dis:
            params = list(self.method.discriminator.parameters())
            optimizer_args = dict(
                self.config.optimizers[self.config.train.disc_optimizer]
            )
            optimizer_args["params"] = params
            self.disc_optimizer = optimizers[self.config.train.disc_optimizer](
                **optimizer_args
            )

            if self.config.model.checkpoint_path != "":
                if "disc_opt" in ckpt.keys():
                    self.disc_optimizer.load_state_dict(ckpt["disc_opt"])
                else:
                    print('WARNING, continuing training without loading disc optimizer state!')

    def _setup_loss(self):
        enc_losses_dict = self.config.encoder_losses
        disc_losses_dict = self.config.disc_losses

        self.loss_builder = LossBuilder(
            enc_losses_dict, 
            disc_losses_dict, 
            self.device
        )

    def _setup_experiment_dir(self):
        base_root = Path(__file__).resolve().parent.parent
        num = 0
        exp_dir = self.config.exp.exp_dir
        exp_dir_name = "{}_{}".format(self.config.exp.name, str(num).zfill(3))

        exp_path = base_root / exp_dir / exp_dir_name
        while True:
            if exp_path.exists():
                num += 1
                exp_dir_name = "{}_{}".format(self.config.exp.name, str(num).zfill(3))
                print(exp_path, "already exists: move to", exp_dir_name)
            else:
                break
            exp_path = base_root / exp_dir / exp_dir_name
        self.experiment_dir = str(exp_path)
        os.makedirs(self.experiment_dir)
        print("Experiment directory: {self.experiment_dir}")

        with open(os.path.join(self.experiment_dir, "config.yaml"), "w") as f:
            omegaconf.OmegaConf.save(config=self.config, f=f.name)

        with open(os.path.join(self.experiment_dir, "run_command.sh"), "w") as f:
            f.write(" ".join(sys.argv))
            f.write("\n")

        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        os.mkdir(self.metrics_dir)
        self.inference_results_dir = os.path.join(
            self.experiment_dir, "inference_results"
        )
        os.mkdir(self.inference_results_dir)

    def _setup_metrics(self):
        metrics_names = self.config.train.val_metrics

        self.metrics = []
        for metric_name in metrics_names:
            metric_args = {}
            if hasattr(self.config.metrics, metric_name):
                metric_args = getattr(self.config.metrics, metric_name)
            self.metrics.append(metrics_registry[metric_name](**metric_args))


    def to_train(self):
        self.method.train()

    def to_eval(self):
        self.method.eval()

    def run(self):
        iter_info = StreamingMeans()
        self.to_train()

        for self.global_step in range(self.start_step, self.config.train.steps + 1):
            with Timer(iter_info, "iter_train"):
                loss_dict = self.train_step()
                iter_info.update({f"iter_train/{k}": v for k, v in loss_dict.items()})

            if self.global_step % self.config.train.val_step == 0 :
                with Timer(iter_info, "iter_val"):
                    val_loss_dict = self.validate()
                    iter_info.update({f"iter_val/{k}": v for k, v in val_loss_dict.items()})

                    orig_pics, method_pics, captions = self.inference_special()
                    self.logger.save_validation_logs(
                        orig_pics,
                        method_pics, 
                        captions, 
                        special_paths=self.special_paths
                    )

            if self.global_step % self.config.train.log_step == 0:
                self.logger.save_train_logs(iter_info, self.global_step)
                iter_info.clear()

            if self.global_step % self.config.train.checkpoint_step == 0:
                self.save_checkpoint()

    def train_step(self):
        x  = next(self.train_dataloader)
        x = x.to(self.device).float()
        output = self.forward(x)

        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])

        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)

        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step:
                print("Start training with discriminator")
            if self.train_dataloader.batch_size != self.config.model.batch_size:
                print(f"Changing batch size from {self.train_dataloader.batch_size} to {self.config.model.batch_size}")
                self.setup_dataloaders(self.config.model.batch_size)

            toogle_grad(self.method.discriminator, True)
            self.method.discriminator.train()

            disc_loss, disc_losses_dict = self.loss_builder.disc_loss(
                self.method.discriminator, 
                output["to_disc"]
                )
            loss_dict.update(disc_losses_dict)

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            toogle_grad(self.method.discriminator, False)
            self.method.discriminator.eval()

        self.method.latent_avg = self.method.latent_avg.detach()

        return loss_dict

    def save_checkpoint(self):
        save_name = f"iteration_{self.global_step}.pt"
        checkpoint_path = os.path.join(self.experiment_dir, save_name)
        save_dict = self.get_save_dict()
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(save_dict, checkpoint_path)

        options_path = os.path.join(self.experiment_dir, "save_options.json")
        save_options = {"start_step": self.global_step + 1}

        if self.config.exp.wandb:
            save_options.update(self.logger.wandb_logger.wandb_args)

        with open(options_path, "w") as f:
            json.dump(save_options, f)

    def get_save_dict(self):
        save_dict = {
            "state_dict": self.method.state_dict(),
            "encoder_opt": self.encoder_optimizer.state_dict(),
            "latent_avg": self.method.latent_avg
        }

        if self.config.train.train_dis:
            save_dict["disc_opt"] = self.disc_optimizer.state_dict()
        return save_dict

    @torch.inference_mode()
    def inference_special(self):
        print("Runing inversion for special")
        self.validate(special=True)

        captions = defaultdict(str)
        for metric in self.metrics:
            if metric.get_name() == "FID":
                continue

            from_data_arg = {
                "fake_data": self.val_pics_res,
                "inp_data": self.val_pics_orig,
                "paths": self.special_paths,
            }
            metric_data, _, _ = metric(
                None, None, out_path=None, from_data=from_data_arg
            )
            for path in self.special_paths:
                metric_value = metric_data[os.path.basename(path)]
                captions[path] += f"{metric.get_name()}: {metric_value:.3}\n"

        return self.val_pics_orig, self.val_pics_res, captions

    @torch.inference_mode()
    def validate(self, special=False):
        if not special:
            print("Start validating")

        self.to_eval()
        self.val_pics_res = []
        self.val_pics_orig = []

        if not special:
            dataloader = self.test_dataloader
            paths = self.paths
        else:
            dataloader = self.special_dataloader
            paths = self.special_paths

        global_i = 0
        for input_batch in tqdm(dataloader):
            input_batch = input_batch.to(self.device).float()
            result_batch = self._run_on_batch(input_batch)
                
            for i in range(result_batch.shape[0]):
                result = tensor2im(result_batch[i])
                img = Image.fromarray(np.array(result)).convert("RGB")

                memory_tmp = BytesIO()
                img.save(memory_tmp, format="jpeg")
                img = Image.open(memory_tmp).convert("RGB")
                memory_tmp.close()

                self.val_pics_res.append(img)
                self.val_pics_orig.append(
                    Image.open(paths[global_i]).convert("RGB")
                )

                global_i += 1

        metrics_dict = {}
        if not special:
            for metric in self.metrics:
                from_data_arg = {
                    "fake_data": self.val_pics_res,
                    "inp_data": self.val_pics_orig,
                    "paths": paths,
                }
                _, metric_mean, _ = metric(
                    None, None, out_path=None, from_data=from_data_arg
                )
                metrics_dict[metric.get_name()] = metric_mean

        self.to_train()
        return metrics_dict

    @abstractmethod
    def _run_on_batch(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()


@training_runners.add_to_registry(name="fse_inverter")
class FSEInverterTrainingRunner(BaseTrainingRunner):
    def forward(self, x):
        y_hat_inv, w_inv, fused_feat, w_feat = self.method(
            x,
            return_latents=True,
            n_iter=self.global_step
        )
                                                
        y_hat_inv_w, _ = self.method.decoder(
            [w_inv],
            input_is_latent=True,
            is_stylespace=False,
            randomize_noise=False
        )

        y_hat = torch.cat([y_hat_inv, y_hat_inv_w], dim=0)
                                   
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        if use_adv_loss:
            output["encoder"]["fake_preds"] = self.method.discriminator(y_hat, None)
            output["to_disc"]["y_hat"] = y_hat
            output["to_disc"]["x"] = x
            output["to_disc"]["step"] = self.global_step
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        x = torch.cat([x, x], dim=0)
        
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat
        output["encoder"]["feat_recon"] = fused_feat
        output["encoder"]["feat_real"] = w_feat

        return output

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch


@training_runners.add_to_registry(name="fse_editor")
class FSEEditorTrainingRunner(BaseTrainingRunner):
    def forward(self, x):
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            d, strenght = get_random_edit()
            
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

            w_e4e = self.method.e4e_encoder(x_resh)
            w_e4e = w_e4e + self.method.latent_avg
            x_E, fx_e4e = self.method.decoder(
                [w_e4e],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            edited_w_e4e = self.get_edited_latent(w_e4e, d, [strenght])
            if isinstance(edited_w_e4e, tuple):
                # stylespace case
                y_E, fy_e4e = self.method.decoder(
                    edited_w_e4e, 
                    is_stylespace=True, 
                    input_is_latent=True,
                    randomize_noise=False,
                    return_features=True
                )
            else:
                edited_w_e4e = torch.cat(edited_w_e4e, dim=0)
                y_E, fy_e4e = self.method.decoder(
                    [edited_w_e4e], 
                    is_stylespace=False,
                    input_is_latent=True,
                    randomize_noise=False,
                    return_features=True
                )

            y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
            delta = fx_e4e[9] - fy_e4e[9]

            if d in self.config.train.disc_edits:
                x_E_256 = torch.cat([x_E_256, x_resh], dim=0)
                delta = torch.cat([delta, delta], dim=0)
            
            
            w_x_E, x_E_predicted_feats = self.method.inverter.fs_backbone(x_E_256)
            w_x_E = w_x_E + self.method.latent_avg
            
            w_x_E_edited = self.get_edited_latent(w_x_E, d, [strenght])
            is_stylespace = isinstance(w_x_E_edited, tuple)
            if not is_stylespace:
                w_x_E_edited = [torch.cat(w_x_E_edited, dim=0)]
            
            _, x_E_w_feats = self.method.decoder(
                [w_x_E],
                input_is_latent=True,
                return_features=True,
                is_stylespace=False,
                randomize_noise=False,
                early_stop=64
            )
            x_E_w_feat = x_E_w_feats[9] 
            to_fuser = torch.cat([x_E_predicted_feats, x_E_w_feat], dim=1)
            x_E_fused_feat = self.method.inverter.fuser(to_fuser)
        

        to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
        x_E_edited_feat = self.method.encoder(to_feature_editor)
        x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)
        

        y_hat_edit, _ = self.method.decoder(
            w_x_E_edited,
            input_is_latent=True,
            new_features=x_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=is_stylespace,
            randomize_noise=False
        )
        
        bs = x_resh.size(0)
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        if use_adv_loss:
            if x_E_256.size(0) > x_resh.size(0):
                assert y_hat_edit.size(0) == bs * 2
                output["encoder"]["fake_preds"] = self.method.discriminator(
                    torch.cat([y_hat_inv, y_hat_edit[bs:]], dim=0), 
                    None
                )
            else:
                output["encoder"]["fake_preds"] = self.method.discriminator(y_hat_inv, None)
            output["to_disc"]["y_hat"] = y_hat_inv
            output["to_disc"]["x"] = x
            output["to_disc"]["step"] = self.global_step
        
        if x_E_256.size(0) > x_resh.size(0):
            assert y_hat_edit.size(0) == bs * 2
            y_hat_edit = y_hat_edit[:bs]

        x = torch.cat([x, y_E], dim=0)
        y_hat = torch.cat([y_hat_inv, y_hat_edit])
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        return output

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch
