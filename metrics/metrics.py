import os
import json
import sys
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_fid.fid_score import calculate_fid_given_paths
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from configs.paths import DefaultPaths

import math
import time
import piq

from PIL import Image

from utils.class_registry import ClassRegistry
from criteria.lpips.lpips import LPIPS
from criteria.id_vit_loss import IDVitLoss
from criteria.id_loss import IDLoss
from criteria.ms_ssim import MSSSIM
from datasets.datasets import FIDDataset, MetricsPathsDataset, MetricsDataDataset

from models.mtcnn.mtcnn import MTCNN
from models.psp.encoders.model_irse import IR_101


metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="lpips")
class LPIPSMetric:
    def __init__(self, batch_size=4, n_workers=4):
        self.loss_func = LPIPS(net_type="alex")
        self.batch_size = batch_size
        self.n_workers = n_workers

    def get_name(self):
        return "LPIPS"

    def __call__(
        self,
        real_data_path,
        fake_data_path=None,
        out_path=None,
        from_data=None,
        silent=False,
    ):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        if from_data:
            dataset = MetricsDataDataset(
                from_data["paths"],
                from_data["inp_data"],
                from_data["fake_data"],
                transform=transform,
            )
        else:
            dataset = MetricsPathsDataset(
                root_path=fake_data_path, gt_dir=real_data_path, transform=transform
            )

        if self.batch_size > len(dataset):
            print(
                (
                    f"Warning: batch size ({self.batch_size}) is bigger than the data size ({len(dataset)}). "
                    "Setting batch size to data size"
                )
            )
            self.batch_size = len(dataset)

        assert self.batch_size > 0

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=False,
        )
        scores_dict = {}
        idx = 0
        for fake_batch, real_batch in tqdm(dataloader):
            for i in range(fake_batch.size(0)):
                loss = float(
                    self.loss_func(
                        fake_batch[i : i + 1].cuda(), real_batch[i : i + 1].cuda()
                    )
                )
                img_path = dataset.paths[idx]
                scores_dict[os.path.basename(img_path)] = loss
                idx += 1

        all_losses = list(scores_dict.values())
        mean_score = np.mean(all_losses)
        std_score = np.std(all_losses)
        if not silent:
            result_str = (
                f"Average {self.get_name()} loss is {mean_score:.3f}+-{std_score:.3f}"
            )
            print(result_str)

        if out_path:
            with open(out_path, "w") as f:
                json.dump(scores_dict, f)
        return scores_dict, mean_score, std_score


@metrics_registry.add_to_registry(name="id_vit")
class ID_VITMetric:
    def __init__(self, batch_size=4, n_workers=4):
        self.loss_func = IDVitLoss()
        self.batch_size = batch_size
        self.n_workers = n_workers

    def get_name(self):
        return "ID_VIT"

    def __call__(
        self,
        real_data_path,
        fake_data_path=None,
        out_path=None,
        from_data=None,
        silent=False,
    ):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        if from_data:
            dataset = MetricsDataDataset(
                from_data["paths"],
                from_data["inp_data"],
                from_data["fake_data"],
                transform=transform,
            )
        else:
            dataset = MetricsPathsDataset(
                root_path=fake_data_path, gt_dir=real_data_path, transform=transform
            )

        if self.batch_size > len(dataset):
            print(
                (
                    f"Warning: batch size ({self.batch_size}) is bigger than the data size ({len(dataset)}). "
                    "Setting batch size to data size"
                )
            )
            self.batch_size = len(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=False,
        )
        scores_dict = {}
        idx = 0
        for fake_batch, real_batch in tqdm(dataloader):
            for i in range(fake_batch.size(0)):
                loss = float(
                    self.loss_func(
                        fake_batch[i : i + 1].cuda(), real_batch[i : i + 1].cuda()
                    )
                )
                img_path = dataset.paths[idx]
                scores_dict[os.path.basename(img_path)] = loss
                idx += 1

        all_losses = list(scores_dict.values())
        mean_score = np.mean(all_losses)
        std_score = np.std(all_losses)
        if not silent:
            result_str = (
                f"Average {self.get_name()} loss is {mean_score:.3f}+-{std_score:.3f}"
            )
            print(result_str)

        if out_path:
            with open(out_path, "w") as f:
                json.dump(scores_dict, f)
        return scores_dict, mean_score, std_score


@metrics_registry.add_to_registry(name="l2")
class L2Metric:
    def __init__(self, batch_size=4, n_workers=4):
        self.loss_func = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.n_workers = n_workers

    def get_name(self):
        return "L2"

    def __call__(
        self,
        real_data_path,
        fake_data_path,
        out_path=None,
        from_data=None,
        silent=False,
    ):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if from_data:
            dataset = MetricsDataDataset(
                from_data["paths"],
                from_data["inp_data"],
                from_data["fake_data"],
                transform=transform,
            )
        else:
            dataset = MetricsPathsDataset(
                root_path=fake_data_path, gt_dir=real_data_path, transform=transform
            )
        if self.batch_size > len(dataset):
            print(
                (
                    "Warning: batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            self.batch_size = len(dataset) 

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=False,
        )
        scores_dict = {}
        idx = 0
        for fake_batch, real_batch in tqdm(dataloader):
            for i in range(fake_batch.size(0)):
                loss = float(
                    self.loss_func(
                        fake_batch[i : i + 1].cuda(), real_batch[i : i + 1].cuda()
                    )
                )
                img_path = dataset.paths[idx]
                scores_dict[os.path.basename(img_path)] = loss
                idx += 1

        all_losses = list(scores_dict.values())
        mean_score = np.mean(all_losses)
        std_score = np.std(all_losses)

        if not silent:
            result_str = (
                f"Average {self.get_name()} loss is {mean_score:.3f}+-{std_score:.3f}"
            )
            print(result_str)

        if out_path:
            with open(out_path, "w") as f:
                json.dump(scores_dict, f)

        return scores_dict, mean_score, std_score


@metrics_registry.add_to_registry(name="fid")
class FIDMetric:
    def __init__(self, batch_size=64, device="cuda", dims=2048, n_workers=8):
        self.batch_size = batch_size
        self.device = device
        self.dims = dims
        self.n_workers = n_workers

    def get_name(self):
        return "FID"

    def __call__(
        self,
        real_data_path,
        fake_data_path,
        out_path=None,
        from_data=None,
        silent=False,
    ):
        if from_data:
            fid_value = self.calculate_fid_given_data(
                from_data, self.batch_size, self.device, self.dims, self.n_workers
            )
        else:
            fid_value = calculate_fid_given_paths(
                [real_data_path, fake_data_path],
                self.batch_size,
                self.device,
                self.dims,
                self.n_workers,
            )

        if not silent:
            result_str = f"Average {self.get_name()} loss is {fid_value:.3f}\n"
            print(result_str)

        if out_path:
            with open(out_path, "w") as f:
                f.write(result_str)
        return None, fid_value, 0.0

    def calculate_fid_given_data(
        self, from_data, batch_size, device, dims, num_workers=1
    ):

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)

        m1, s1 = self.calculate_activation_statistics(
            from_data["inp_data"], model, batch_size, dims, device, num_workers, 
        )
        m2, s2 = self.calculate_activation_statistics(
            from_data["fake_data"], model, batch_size, dims, device, num_workers,
        )

        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value

    def calculate_activation_statistics(
        self, files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, fid_func=None, is_real=False
    ):
        act = self.get_activations(files, model, batch_size, dims, device, num_workers, fid_func=fid_func, is_real=is_real)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def get_activations(
        self, files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, fid_func=None, is_real=False
    ):
        model.eval()

        if batch_size > len(files):
            print(
                (
                    "Warning: batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            batch_size = len(files)

        if np.all(files[0].size == files[1].size):
            dataset = FIDDataset(files, transforms=transforms.ToTensor())
        else:
            # cars case
            dataset = FIDDataset(files, transforms=transforms.Compose([
                transforms.Resize((384, 512)),
                transforms.ToTensor()]))

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        pred_arr = np.empty((len(files), dims))
        start_idx = 0
        face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))

        for batch in tqdm(dataloader):
            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx : start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
        return pred_arr

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@metrics_registry.add_to_registry(name="msssim")
class MSSSIMMetric:
    def __init__(self):
        self.loss_func = piq.multi_scale_ssim
        self.image_size = (1024, 1024)

    def get_name(self):
        return "MSSSIM"

    def __call__(
        self,
        real_data_path,
        fake_data_path,
        out_path=None,
        from_data=None,
        silent=False,
    ):

        if from_data:
            scores_dict = {}
            image_transform = transforms.ToTensor()
            cnt = 0

            for real_image, fake_image in tqdm(
                zip(from_data["inp_data"], from_data["fake_data"])
            ):
                gt_image = real_image.convert("RGB").resize(self.image_size)
                gt_image = image_transform(gt_image).cuda().unsqueeze(0)

                pred_image = fake_image.convert("RGB").resize(self.image_size)
                pred_image = image_transform(pred_image).cuda().unsqueeze(0)

                score = self.loss_func(pred_image, gt_image, data_range=1.0)
                scores_dict[os.path.basename(from_data["paths"][cnt])] = score.item()
                cnt += 1
        else:
            real_filenames = sorted(os.listdir(real_data_path))
            fake_filenames = sorted(os.listdir(fake_data_path))

            scores_dict = {}
            image_transform = transforms.ToTensor()

            for real_fn, fake_fn in tqdm(zip(real_filenames, fake_filenames)):
                gt_image = (
                    Image.open(os.path.join(real_data_path, real_fn))
                    .convert("RGB")
                    .resize(self.image_size)
                )
                gt_image = image_transform(gt_image).cuda().unsqueeze(0)

                pred_image = (
                    Image.open(os.path.join(fake_data_path, fake_fn))
                    .convert("RGB")
                    .resize(self.image_size)
                )
                pred_image = image_transform(pred_image).cuda().unsqueeze(0)

                score = self.loss_func(pred_image, gt_image, data_range=1.0)
                scores_dict[os.path.basename(real_fn)] = score.item()
        all_losses = list(scores_dict.values())
        mean_score = np.mean(all_losses)
        std_score = np.std(all_losses)

        if not silent:
            result_str = (
                f"Average {self.get_name()} loss is {mean_score:.3f}+-{std_score:.3f}"
            )
            print(result_str)

        if out_path:
            with open(out_path, "w") as f:
                json.dump(scores_dict, f)

        return scores_dict, mean_score, std_score


@metrics_registry.add_to_registry(name="id")
class IDMetric:
    def __init__(
        self,
        n_threads=8,
    ):  
        self.curricular_face_path = DefaultPaths.curricular_face_path
        self.n_threads = n_threads
        try:
          torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
          pass 

    def get_name(self):
        return "ID"

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _extract_on_paths(self, file_paths):
        facenet = IR_101(input_size=112)
        facenet.load_state_dict(torch.load(self.curricular_face_path))
        facenet.cuda()
        facenet.eval()
        mtcnn = MTCNN()
        id_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        pid = mp.current_process().name
        tot_count = len(file_paths)
        count = 0

        scores_dict = {}
        for res_path, gt_path in file_paths:
            count += 1
            if True:
                input_im = Image.open(res_path)
                input_im, _ = mtcnn.align(input_im)
                if input_im is None:
                    print("{} skipping {}".format(pid, res_path))
                    continue

                input_id = facenet(id_transform(input_im).unsqueeze(0).cuda())[0]

                result_im = Image.open(gt_path)
                result_im, _ = mtcnn.align(result_im)
                if result_im is None:
                    print("{} skipping {}".format(pid, gt_path))
                    continue

                result_id = facenet(id_transform(result_im).unsqueeze(0).cuda())[0]
                score = float(input_id.dot(result_id))
                scores_dict[os.path.basename(gt_path)] = score
        return scores_dict

    def extract_on_data(self, inp):
        inp_data, fake_data, paths = inp
        inp_data = [inp_data]
        fake_data = [fake_data]
        paths = [paths]

        facenet = IR_101(input_size=112)
        facenet.load_state_dict(torch.load(self.curricular_face_path))
        facenet.cuda()
        facenet.eval()
        mtcnn = MTCNN()
        id_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        pid = mp.current_process().name
        tot_count = len(paths)
        count = 0

        scores_dict = {}
        for i in range(len(paths)):
            count += 1

            input_im = inp_data[i]
            input_im = Image.open(paths[i])

            input_im, _ = mtcnn.align(input_im)
            if input_im is None:
                print("{} skipping {}".format(pid, paths[i]))
                continue

            input_id = facenet(id_transform(input_im).unsqueeze(0).cuda())[0]

            result_im = fake_data[i]

            result_im, _ = mtcnn.align(result_im)
            if result_im is None:
                print("{} skipping {}".format(pid, paths[i]))
                continue

            result_id = facenet(id_transform(result_im).unsqueeze(0).cuda())[0]
            score = float(input_id.dot(result_id))
            scores_dict[os.path.basename(paths[i])] = score
        return scores_dict

    def __call__(
        self,
        real_data_path,
        fake_data_path,
        out_path=None,
        from_data=None,
        silent=False,
    ):

        pool = mp.Pool(self.n_threads)
        if from_data:
            zipped = zip(
                from_data["inp_data"], from_data["fake_data"], from_data["paths"]
            )
            results = pool.map(self.extract_on_data, zipped)
        else:
            file_paths = []
            for f in tqdm(os.listdir(fake_data_path)):
                image_path = os.path.join(fake_data_path, f)
                gt_path = os.path.join(real_data_path, f)
                if f.endswith(".jpg") or f.endswith(".png"):
                    file_paths.append([image_path, gt_path.replace(".png", ".jpg")])

            file_chunks = list(
                self._chunks(
                    file_paths, int(math.ceil(len(file_paths) / self.n_threads))
                )
            )
            results = pool.map(self._extract_on_paths, file_chunks)

        scores_dict = {}
        for d in results:
            scores_dict.update(d)

        all_scores = list(scores_dict.values())
        mean = np.mean(all_scores)
        std = np.std(all_scores)

        if not silent:
            result_str = "New ID Average score is {:.3f}+-{:.3f}".format(mean, std)
            print(result_str)

        if out_path:
            with open(out_path, "w") as f:
                json.dump(scores_dict, f)

        return scores_dict, mean, std
        