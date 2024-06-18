import torch
from torch import nn
import torch.nn.functional as F
from configs.paths import DefaultPaths


class MocoLoss(nn.Module):
    def __init__(self):
        super(MocoLoss, self).__init__()
        print("Loading MOCO model from path: {}".format(DefaultPaths.moco))
        self.model = self.__load_model()
        self.model.cuda()
        self.model.eval()

    @staticmethod
    def __load_model():
        import torchvision.models as models

        model = models.__dict__["resnet50"]()
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False
        checkpoint = torch.load(DefaultPaths.moco, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # remove output layer
        model = nn.Sequential(*list(model.children())[:-1]).cuda()
        return model

    def extract_feats(self, x):
        x = F.interpolate(x, size=224)
        x_feats = self.model(x)
        x_feats = nn.functional.normalize(x_feats, dim=1)
        x_feats = x_feats.squeeze()
        return x_feats
        

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count
