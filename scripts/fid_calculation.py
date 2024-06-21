import sys
import torch

sys.path =  ['.'] + sys.path

from argparse import ArgumentParser
from pathlib import Path
from metrics.metrics import metrics_registry
from datasets.transforms import transforms_registry
from datasets.datasets import CelebaAttributeDataset
from utils.common_utils import tensor2im, setup_seed


setup_seed(777)


def inferece_fid_editing(opts):
    fid_metric = metrics_registry["fid"]()
    transform = transforms_registry[opts.transforms]().get_transforms()["test"]

    attr_name = opts.attr_name

    attr_dataset = CelebaAttributeDataset(
        opts.orig_path, 
        attr_name,
        transform, 
        opts.celeba_attr_table_pth, 
        use_attr=not opts.attr_is_reversed
    )

    not_attr_dataset = CelebaAttributeDataset(
        opts.synt_path, 
        attr_name,
        transform, 
        opts.celeba_attr_table_pth, 
        use_attr=opts.attr_is_reversed
        )

    print(f"Percent of Images of attribute {opts.attr_name} is "
                  f"{len(attr_dataset) / (len(attr_dataset) + len(not_attr_dataset))}")

    attr_images = []
    for attr_image in attr_dataset:
        img = tensor2im(attr_image).convert("RGB")
        attr_images.append(img)

    edited_images = []
    for not_attr_image in not_attr_dataset:
        img = tensor2im(not_attr_image).convert("RGB")
        edited_images.append(img)

    from_data_arg = {
        "inp_data": attr_images,
        "fake_data": edited_images,
        "paths": [],
    }

    _, fid_value, _ = fid_metric("", "", out_path="", from_data=from_data_arg)
    print(f"FID for {opts.attr_name} is {fid_value:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--orig_path", type=str, help="Path to directory of original Celeba images "
    )
    parser.add_argument(
        "--synt_path",
        type=str,
        help="Path to synthesized edited images",
    )
    parser.add_argument(
        "--attr_name",
        type=str,
        help="Name of Celeba attribute that is added during editing.",
    )
    parser.add_argument(
        "--attr_is_reversed",
        action='store_true',
        help="Means that attribute was not added but removed during editing",
    )
    parser.add_argument(
        "--celeba_attr_table_pth",
        default="CelebAMask-HQ-attribute-anno.txt",
        type=str,
        help="Path to celeba attributes .txt",
    )
    parser.add_argument(
        "--transforms",
        default="face_1024",
        type=str,
        help="Which transforms from datasets.transforms.transforms_registry should be used",
    )

    opts = parser.parse_args()
    inferece_fid_editing(opts)
