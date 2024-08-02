import os
import cv2
import PIL
import torch
import subprocess
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path

from omegaconf import OmegaConf
from utils.common_utils import tensor2im, tensor2im_no_tfm, MaskerCantFindFaceError
from datasets.transforms import transforms_registry
from runners.inference_runners import FSEInferenceRunner


def extract_mask(image_path, save_dir_path, trash=0.995):
    from models.farl.farl import Masker

    save_dir_path = Path(save_dir_path)
    image_path = Path(image_path)

    orig_img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    orig_img_tensor = transform(orig_img)

    orig_img_tensor = (orig_img_tensor.unsqueeze(0) * 255).long().cuda()

    with torch.inference_mode():
        # try to find trashhlod for detecting face
        for detector_trash in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]:
            masker = Masker(trash=detector_trash)
            faces = masker.face_detector(orig_img_tensor)
            if len(faces['image_ids']) != 0:
                break

        if len(faces['image_ids']) == 0:
            raise MaskerCantFindFaceError("Masker's face detector can't find face in your image 😢")
        faces = masker.face_parser(orig_img_tensor, faces)


    background_mask = F.sigmoid(faces['seg']['logits'][:, 0])
    background_mask = background_mask[0].unsqueeze(0)
    background_mask = (background_mask >= trash).cpu()
    mask_path = save_dir_path / (image_path.stem + "_mask.jpg")
    to_save = (background_mask[0] * 255).long().numpy()
    mask = Image.fromarray(to_save.astype(np.uint8)).convert("1")
    mask.save(mask_path)

    backfround_tens = orig_img_tensor[0].cpu() / 255 * background_mask.float().repeat(3, 1, 1)
    background = tensor2im_no_tfm(backfround_tens)
    back_path = save_dir_path / (image_path.stem + "_back.jpg")
    background.save(back_path)

    face_tens = orig_img_tensor[0].cpu() / 255 * (1 - background_mask.float()).repeat(3, 1, 1)
    face = tensor2im_no_tfm(face_tens)
    face_path = save_dir_path / (image_path.stem + "_face.jpg")
    face.save(face_path)

    return mask_path


def run_alignment(image_path):
  import dlib
  from scripts.align_all_parallel import align_face

  predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
  aligned_image, unalign_dict = align_face(filepath=image_path, predictor=predictor)
  return aligned_image, unalign_dict


def unalign(edited_image, unalign_dict, orig_img_pth, unaligned_path):
    quad = unalign_dict["quad"]
    source_quad = [(0, 0), (1024, 0),  (1024, 1024), (0, 1024)]
    dest_quad = np.array([quad[3], quad[0], quad[1], quad[2]])
    M = cv2.getPerspectiveTransform(dest_quad.astype(np.float32), np.array(source_quad).astype(np.float32))
    unaligned = edited_image.transpose(PIL.Image.FLIP_LEFT_RIGHT).transform(unalign_dict["pretrans_size"], PIL.Image.PERSPECTIVE, M.reshape(-1), PIL.Image.BILINEAR)

    mask = np.asarray(unaligned) > 0
    mask = np.stack([mask[:,:,0] | mask[:,:,1] | mask[:,:,2]] * 3, axis=-1)

    if "blur1" in unalign_dict:
      unaligned -= unalign_dict["blur2"]
      unaligned -= unalign_dict["blur1"]
      pad = unalign_dict["pad"]
      unaligned = PIL.Image.fromarray(np.uint8(np.clip(np.rint(unaligned), 0, 255)), 'RGB').crop([pad[1], pad[0],  unaligned.shape[1] - pad[3], unaligned.shape[0] - pad[2]])
      mask = mask[pad[0]:mask.shape[0]-pad[1], pad[2]:mask.shape[1]-pad[3]]

    img_orig = PIL.Image.open(orig_img_pth).convert("RGB")

    if "crop" in unalign_dict:
      crop = unalign_dict["crop"]
      unaligned = np.pad(np.float32(unaligned), ((crop[1], img_orig.size[1] - crop[3]), (crop[0], img_orig.size[0] - crop[2]), (0, 0)))
      mask = np.pad(np.float32(mask), ((crop[1], img_orig.size[1] - crop[3]), (crop[0], img_orig.size[0] - crop[2]), (0, 0)))
      unaligned = PIL.Image.fromarray(np.uint8(np.clip(np.rint(unaligned), 0, 255)), 'RGB')

    if "shrink" in unalign_dict:
      unaligned = unaligned.resize(unalign_dict["shrink"])
      mask = mask.resize(unalign_dict["shrink"])

    unaligned = np.asarray(img_orig) * (1 - mask / mask.max()) + np.asarray(unaligned) * mask / mask.max()
    PIL.Image.fromarray(unaligned.astype('uint8'), 'RGB').save("edited.png")
    PIL.Image.fromarray(np.uint8(np.clip(np.rint((1 - mask) * 255), 0, 255)), 'RGB').save("mask.jpg")

    subprocess.run(
            ["fpie", "-s", orig_img_pth, "-m", "mask.jpg", "-t", "edited.png", "-o", unaligned_path, "-n",
             "5000", "-b", "taichi-gpu", "-g", "src"],
            check=True
        )


class SimpleRunner:
    def __init__(
        self, 
        editor_ckpt_pth: str, 
        simple_config_pth: str = "configs/simple_inference.yaml"
    ):

        config = OmegaConf.load(simple_config_pth)
        config.model.checkpoint_path = editor_ckpt_pth
        config.methods_args.fse_full = {}

        self.inference_runner = FSEInferenceRunner(config)
        self.inference_runner.setup()
        self.inference_runner.method.eval()
        self.inference_runner.method.decoder = self.inference_runner.method.decoder.float()

    def edit(
        self,
        orig_img_pth: str,
        editing_name: str,
        edited_power: float,
        save_pth: str,
        align: bool = False,
        use_mask: bool = False,
        mask_trashold=0.995,
        mask_path: str = None,
        save_e4e=False,
        save_inversion=False
    ):

        save_pth = Path(save_pth)
        save_pth_dir = save_pth.parents[0]
        save_pth_dir.mkdir(parents=True, exist_ok=True)
        aligned_image_pth = orig_img_pth

        if align:
            aligned_image, unalign_dict = run_alignment(orig_img_pth)
            save_align_pth = save_pth.parents[0] / (save_pth.stem + "_aligned.jpg")
            print(f"Save aligned image to {save_align_pth}")
            aligned_image.convert('RGB').save(save_align_pth)
            aligned_image_pth = save_align_pth

        if use_mask and mask_path is None:
            print("Prepearing mask")
            mask_path = extract_mask(aligned_image_pth, save_pth.parents[0], trash=mask_trashold)
            print("Done")

        if use_mask and mask_path is not None:
            print(f"Use mask from {mask_path}")
            mask = Image.open(mask_path).convert("RGB")
            transform = transforms.ToTensor()
            mask = transform(mask).unsqueeze(0).to(self.inference_runner.device)
        else:
            mask = None

        orig_img = Image.open(aligned_image_pth).convert("RGB")
        transform_dict = transforms_registry["face_1024"]().get_transforms()
        orig_img = transform_dict["test"](orig_img).unsqueeze(0)

        device = self.inference_runner.device
        inv_images, inversion_results = self.inference_runner._run_on_batch(orig_img.to(device))
        edited_image = self.inference_runner._run_editing_on_batch(
            method_res_batch=inversion_results, 
            editing_name=editing_name, 
            editing_degrees=[edited_power],
            mask=mask,
            return_e4e=save_e4e
        )

        if save_inversion:
            save_inv_pth = save_pth.parents[0] / (save_pth.stem + "_inversion.jpg")
            inv_image = tensor2im(inv_images[0].cpu())
            inv_image.save(save_inv_pth)

        if save_e4e:
            edited_image, e4e_inv, e4e_edit = edited_image

            save_e4e_inv_pth = save_pth.parents[0] / (save_pth.stem + "_e4e_inversion.jpg")
            e4e_inv_image = tensor2im(e4e_inv[0].cpu())
            e4e_inv_image.save(save_e4e_inv_pth)

            save_e4e_edit_pth = save_pth.parents[0] / (save_pth.stem + "_e4e_edit.jpg")
            e4e_edit_image = tensor2im(e4e_edit[0].cpu())
            e4e_edit_image.save(save_e4e_edit_pth)

        edited_image = tensor2im(edited_image[0][0].cpu())
        edited_image.save(save_pth)

        if align:
            unaligned_path = save_pth.parents[0] / (save_pth.stem + "_unaligned.jpg")
            unalign(edited_image, unalign_dict, orig_img_pth, unaligned_path)

        return edited_image

    def available_editings(self):
        edits_types = []
        for field in dir(self.inference_runner.latent_editor):
            if "directions" in field.split("_"):
                edits_types.append(field)

        print("This code handles the following editing directions for following methods:")
        available_directions = {}
        for edit_type in edits_types:
            print(edit_type + ":")
            edit_type_directions = getattr(self.inference_runner.latent_editor, edit_type, None).keys()
            for direction in edit_type_directions:
                print("\t" + direction)
        print(GLOBAL_DIRECTIONS_DESC)

GLOBAL_DIRECTIONS_DESC ="""
You can alse use directions from text prompts via StyleClip Global Mapper (https://arxiv.org/abs/2103.17249).
Such directions look as follows: "styleclip_global_{neutral prompt}_{target prompt}_{disentanglement}" where
neutral prompt -- some neutral description of the original image (e.g. "a face")
target prompt -- text that contains the desired edit (e.g. "a smilling face")
disentanglement -- positive number, the more this attribute - the more related attributes will also be changed (e.g. 
for grey hair editing, wrinkle, skin colour and glasses may also be edited)

Example: "styleclip_global_face with hair_face with black hair_0.18"

More information about the purpose of directions and their approximate power range can be found in available_directions.txt.
"""
