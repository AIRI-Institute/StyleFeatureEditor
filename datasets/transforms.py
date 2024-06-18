from abc import abstractmethod
import torchvision.transforms as transforms
from utils.class_registry import ClassRegistry

transforms_registry = ClassRegistry()


class TransformsConfig(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_transforms(self):
        pass

class FaceTransforms(TransformsConfig):
    def __init__(self):
        super(FaceTransforms, self).__init__()
        self.image_size = None

    def get_transforms(self):
        transforms_dict = {
            "train": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        }
        return transforms_dict


@transforms_registry.add_to_registry(name="face_256")
class Face256Transforms(FaceTransforms):
    def __init__(self):
        super(Face256Transforms, self).__init__()
        self.image_size = (256, 256)


@transforms_registry.add_to_registry(name="face_1024")
class Face1024Transforms(FaceTransforms):
    def __init__(self):
        super(Face1024Transforms, self).__init__()
        self.image_size = (1024, 1024)


