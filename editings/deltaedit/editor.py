import torch
import clip
import copy
import numpy as np
import torch.nn.functional as F

from editings.deltaedit import map_tool
from editings.deltaedit.delta_mapper import DeltaMapper


STYLE_DIM = [512] * 10 + [256, 256, 128, 128, 64, 64, 32]


def GetBoundary(fs3, dt, threshold):
    tmp = np.dot(fs3, dt)
    select = np.abs(tmp) < threshold
    return select

def improved_ds(ds, select):
    ds_imp = copy.copy(ds)
    ds_imp[select] = 0
    ds_imp = ds_imp.unsqueeze(0)
    return ds_imp


class DeltaEditor:
    def __init__(self):
        device = "cuda"
        self.fs3 = np.load("pretrained_models/fs3.npy")
        np.set_printoptions(suppress=True)

        self.net = DeltaMapper()
        net_ckpt = torch.load("pretrained_models/delta_mapper.pt")
        self.net.load_state_dict(net_ckpt)
        self.net = self.net.to(device).eval()

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.upsample = torch.nn.Upsample(scale_factor=7)

    def get_delta_s(self, neutral, target, trash, orig_image, start_s):
        with torch.no_grad():
            classnames = [target, neutral]
            dt = map_tool.GetDt(classnames, self.clip_model)
            select = GetBoundary(self.fs3, dt, trash)
            dt = torch.Tensor(dt).cuda()
            dt = dt / dt.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)

            img_gen_for_clip = self.avg_pool(orig_image)
            c_latents = self.clip_model.encode_image(img_gen_for_clip.cuda())
            c_latents = c_latents / c_latents.norm(dim=-1, keepdim=True).float()

            delta_c = torch.cat((c_latents, dt.unsqueeze(0)), dim=1)
            fake_delta_s = self.net(torch.cat(start_s, dim=-1), delta_c)
            improved_fake_delta_s = improved_ds(fake_delta_s[0], select)
            return torch.split(improved_fake_delta_s, STYLE_DIM, dim=-1)



