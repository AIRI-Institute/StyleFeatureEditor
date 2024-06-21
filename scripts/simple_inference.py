import sys
import torch
import random

sys.path =  ['.'] + sys.path

from argparse import ArgumentParser
from utils.common_utils import setup_seed
from runners.simple_runner import SimpleRunner


setup_seed(777)


def run(opts):
    runner = SimpleRunner(
        editor_ckpt_pth="pretrained_models/sfe_editor_light.pt",
    )

    runner.edit(
        orig_img_pth=opts.orig_img_pth,
        editing_name=opts.editing_name,
        edited_power=opts.edited_power,
        save_pth=opts.save_pth,
        align=opts.align,
        use_mask=opts.use_mask,
        mask_trashold=opts.mask_trashold,
        mask_path=opts.mask_path
    )
    runner.available_editings()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--orig_img_pth", type=str, help="Path to original image"
    )
    parser.add_argument(
        "--editing_name",
        type=str,
        help="Name of desired editing",
    )
    parser.add_argument(
        "--edited_power",
        type=float,
        help="Power of desired editing, float",
    )
    parser.add_argument(
        "--save_pth",
        type=str,
        help="Path where to save edited (and aligned) image",
    )
    parser.add_argument(
        "--align",
        action='store_true',
        help="Flag to align image if it was not",
    )
    parser.add_argument(
        "--use_mask",
        action='store_true',
        help="Flag to edit only masked zone. May be usefull to remove background artefacts.",
    )
    parser.add_argument(
        "--mask_trashold",
        type=float,
        default=0.095,
        help="Trashold in range (0, 1) to separate face from background",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to custom background mask",
    )

    opts = parser.parse_args()
    run(opts)
