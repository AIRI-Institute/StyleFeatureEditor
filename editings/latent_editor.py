import os
import sys
import torch
import pickle
import argparse
import numpy as np

from editings import ganspace
from editings.styleclip.mapper.styleclip_mapper import StyleCLIPMapper
from editings.styleclip.mapper.gloabl_mapper import StyleCLIPGlobalDirection
from editings.deltaedit.editor import DeltaEditor


STYLESPACE_IDX = [
            0,
            1,
            1,
            2,
            2,
            3,
            4,
            4,
            5,
            6,
            6,
            7,
            8,
            8,
            9,
            10,
            10,
            11,
            12,
            12,
            13,
            14,
            14,
            15,
            16,
            16,
        ]


class LatentEditor:
    def __init__(self, domain="human_faces"):

        self.domain = domain

        if self.domain == "human_faces":
            self.interfacegan_directions = {
                "age": "editings/interfacegan_directions/age.pt",
                "smile": "editings/interfacegan_directions/smile.pt",
                "rotation": "editings/interfacegan_directions/rotation.pt",
            }
            self.interfacegan_tensors = {
                name: torch.load(path).cuda()
                for name, path in self.interfacegan_directions.items()
            }

            self.ganspace_pca = torch.load("editings/ganspace_pca/ffhq_pca.pt")
            self.ganspace_directions = {
                "eye_openness": (54, 7, 8, 5),
                "trimmed_beard": (58, 7, 9, 7),
                "lipstick": (34, 10, 11, 20),
                "face_roundness": (37, 0, 5, 20.0),
                "nose_length": (51, 4, 5, -30.0),
                "eyebrow_thickness": (37, 8, 9, 20.0),
                "head_angle_up": (11, 1, 4, -10.5),
                "displeased": (36, 4, 7, 10.0),
            }

            self.styleclip_directions = {
                "afro": [False, False, True],
                "angry": [False, False, True],
                "beyonce": [False, False, False],
                "bobcut": [False, False, True],
                "bowlcut": [False, False, True],
                "curly_hair": [False, False, True],
                "hilary_clinton": [False, False, False],
                "depp": [False, False, False],
                "mohawk": [False, False, True],
                "purple_hair": [False, False, False],
                "surprised": [False, False, True],
                "taylor_swift": [False, False, False],
                "trump": [False, False, False],
                "zuckerberg": [False, False, False],
            }
            self.styleclip_global_editor = self.load_styleclip_global()

            self.stylespace_directions = {
                "black hair": [(12, 479)],
                "blond hair": [(12, 479), (12, 266)],
                "grey hair": [(11, 286)],
                "wavy hair": [(6, 500), (8, 128), (5, 92), (6, 394), (6, 323)],
                "bangs": [
                    (3, 259),
                    (6, 285),
                    (5, 414),
                    (6, 128),
                    (9, 295),
                    (6, 322),
                    (6, 487),
                    (6, 504),
                ],
                "receding hairline": [(5, 414), (6, 322), (6, 497), (6, 504)],
                "smiling": [(6, 501)],
                "sslipstick": [(15, 45)],
                "sideburns": [(12, 237)],
                "goatee": [(9, 421)],
                "earrings": [(8, 81)],
                "glasses": [(3, 288), (2, 175), (3, 120), (2, 97)],
                "wear suit": [(9, 441), (8, 292), (11, 358), (6, 223)],
                "gender": [(9, 6)],
            }
            
            self.fs_directions = {
                "fs_glasses": "editings/bound/Eyeglasses_boundary.npy",
                "fs_smiling": "editings/bound/Smiling_boundary.npy",
                "fs_makeup": "editings/bound/Heavy_Makeup_boundary.npy"
             }

            self.deltaedit_editor = DeltaEditor()

        elif self.domain == "car":

            self.stylespace_directions = {
                "front": [(8, 411)],
                "headlights": [(8, 441), (9, 355)],
                "grill": [(9, 191)],
                "trees": [(9, 108)],
                "grass_ss": [(12, 107)],
                "sky": [(12, 76)],
                "hubcap": [(12, 113), (12, 439)],
                "car color": [(12, 142), (15, 227)],
                "logo": [(9, 185)],
                "wheel angle": [(8, 420)],
            }

            self.ganspace_pca = torch.load("editings/ganspace_pca/cars_pca.pt")
            self.ganspace_directions = {
                "pose_1": (0, 0, 5, 2),
                "pose_2": (0, 0, 5, -2),
                "cube": (16, 3, 6, 25),
                "color": (22, 9, 11, -8),
                "grass": (41, 9, 11, -18)
            }


    def load_styleclip_global(self):
        delta_i_c = torch.from_numpy(np.load("editings/styleclip/global_mapper_data/delta_i_c.npy")).float().cuda()
        with open("editings/styleclip/global_mapper_data/S_mean_std", "rb") as channels_statistics:
            _, s_std = pickle.load(channels_statistics)
            s_std = [torch.from_numpy(s_i).float().cuda() for s_i in s_std]
        with open("editings/styleclip/global_mapper_data/templates.txt", "r") as templates:
            text_prompt_templates = templates.readlines()
        global_direction_calculator = StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates)
        return global_direction_calculator


    def get_styleclip_mapper_edits(self, start_w, factors, direction):
        latents_to_display = []
        mapper_checkpoint_path = os.path.join(
            "pretrained_models/styleclip_mappers",
            f"{direction}.pt",
        )
        ckpt = torch.load(mapper_checkpoint_path, map_location="cpu")
        opts = ckpt["opts"]
        styleclip_opts = argparse.Namespace(
            **{
                "mapper_type": "LevelsMapper",
                "no_coarse_mapper": self.styleclip_directions[direction][0],
                "no_medium_mapper": self.styleclip_directions[direction][1],
                "no_fine_mapper": self.styleclip_directions[direction][2],
                "stylegan_size": 1024,
                "checkpoint_path": mapper_checkpoint_path,
            }
        )
        opts.update(vars(styleclip_opts))
        opts = argparse.Namespace(**opts)
        style_clip_net = StyleCLIPMapper(opts)
        style_clip_net.eval()
        style_clip_net.cuda()
        direction = style_clip_net.mapper(start_w)
        for factor in factors:
            edited_latent = start_w + factor * direction
            latents_to_display.append(edited_latent)

        return latents_to_display


    def get_styleclip_global_edits(self, start_s, factors, direction):
        latents_to_display = []

        neutral_text, target_text, disentanglement = direction.split("_")
        disentanglement = float(disentanglement)

        directions = self.styleclip_global_editor.get_delta_s(neutral_text, target_text, disentanglement)
        factors = torch.tensor(factors).cuda().view(-1, 1)
        srart_ss, start_rgb = start_s

        edits_rgb = []
        edits_ss = []
        for i in range(26):
            if i in [1, 4, 7, 10, 13, 16, 19, 22, 25]:
                edits_rgb.append(directions[i].view(1, -1).repeat(len(factors), 1))
            else:
                edits_ss.append(directions[i].view(1, -1).repeat(len(factors), 1))

        edited_rgb = []
        edited_ss = []
        for orig, edit in zip(srart_ss, edits_ss):
            edited_ss.append(orig.repeat(len(factors), 1) + edit * factors.repeat(1, orig.size(1)) / 1.5)
        for orig, edit in zip(start_rgb, edits_rgb):
            edited_rgb.append(orig.repeat(len(factors), 1) + edit * factors.repeat(1, orig.size(1)) / 1.5)
 

        return edited_ss, edited_rgb

    def get_deltaedit_edits(self, start_s, factors, direction, original_image):
        latents_to_display = []

        neutral_text, target_text, disentanglement = direction.split("_")
        disentanglement = float(disentanglement)

        factors = torch.tensor(factors).cuda().view(-1, 1)
        srart_ss, edited_rgb = start_s
        edits_ss = self.deltaedit_editor.get_delta_s(neutral_text, target_text, disentanglement, original_image, srart_ss)

        edited_rgb = [latent.repeat(len(factors), 1) for latent in edited_rgb]
        edited_ss = []
        for orig, edit in zip(srart_ss, edits_ss):
            edited_ss.append(orig.repeat(len(factors), 1) + edit * factors.repeat(1, orig.size(1)))

        return edited_ss, edited_rgb


    def get_ganspace_edits(self, start_w, factors, direction):
        latents_to_display = []
        for factor in factors:
            ganspace_direction = self.ganspace_directions[direction]
            edit_direction = list(ganspace_direction)
            edit_direction[-1] = factor
            edit_direction = tuple(edit_direction)

            new_w = ganspace.edit(start_w, self.ganspace_pca, [edit_direction])
            latents_to_display.append(new_w)
        return latents_to_display


    def get_interface_gan_edits(self, start_w, factors, direction):
        latents_to_display = []
        for factor in factors:
            tensor_direction = self.interfacegan_tensors[direction]
            edited_latent = start_w + factor / 2 * tensor_direction
            latents_to_display.append(edited_latent)
        return latents_to_display


    def get_stylespace_edits(self, start_s, factors, direction):
        edits = self.stylespace_directions[direction]
        start_stylespaces, start_stylespaces_rgb = start_s
        device = start_stylespaces[0].device
        latents_to_display = []

        edited_latent = [
            s.clone().repeat(len(factors), 1)
            for s in start_stylespaces
        ]
        factors = torch.tensor(factors).to(device)
        for layer_num, feat_num in edits:
            edited_latent[STYLESPACE_IDX[layer_num]][:, feat_num] += factors * 3
        edited_stylespaces_rgb = [
            rgb.repeat(len(factors), 1) for rgb in start_stylespaces_rgb
        ]

        return edited_latent, edited_stylespaces_rgb
        

    def get_fs_edits(self, w, factors, direction):
        path = self.fs_directions[direction]
        boundary = np.load(path)
        device = w.device
        bs = w.size(0)
        w_0 = w.cpu().numpy().reshape(bs, -1)
        boundary = boundary.reshape(1, -1).repeat(bs, 0)

        edits = [torch.tensor(w_0 + factor * boundary).view(bs, -1, 512).to(device) for factor in factors]

        return edits

