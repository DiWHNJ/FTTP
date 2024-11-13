from pathlib import Path
from PIL import Image, ImageDraw
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as torch_transforms
from datasets import load_dataset
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import InterpolationMode
import os
import pdb  
INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}

class_to_name = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
    "n03923666": "Elon Musk",         #10
    "n03923777": "Hello Kitty",
    "n03923888": "Facebook",
    "n03923999": "Tylor Swift",
}
def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize(size, interpolation=interpolation),
            torch_transforms.CenterCrop(size),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform

#  python train-scripts/generate_mask.py --ckpt_path 'models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt' --classes 1 --device '0'
class Imagenette(Dataset):
    def __init__(self, split, class_to_forget=None, transform=None):
        # self.dataset = load_dataset("imagefolder",data_dir="/data2/ljq/stable-diffusion/imagenette", streaming=True,name = "160px")[split]
        self.dataset = load_dataset("imagefolder", data_dir="/data2/ljq/stable-diffusion/imagenette")[split]

        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.dataset.features["label"].names)
        }
        self.file_to_class = {
            str(idx): self.dataset["label"][idx] for idx in range(len(self.dataset))
        }

        self.class_to_forget = class_to_forget
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if example["label"] == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)
        return image, label





# class ImagenetteCustom(Dataset):
#     def __init__(self, root_dir='/data2/ljq/stable-diffusion/imagenette', split='train', class_to_forget=None, transform=None):
#         """
#         root_dir: 数据集根目录
#         split: 'train' 或 'val'，决定加载训练集或验证集
#         class_to_forget: 如果指定了这个类别，标签会被随机替换
#         transform: 数据变换操作
#         """
#         self.root_dir = os.path.join(root_dir, split)
#         self.class_to_forget = class_to_forget
#         self.transform = transform
#         # import pdb
#         # pdb.set_trace()
#         # 获取类名（文件夹名）及对应的标签
#         self.classes = sorted(os.listdir(self.root_dir))  # 文件夹名字即类别名
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

#         # 获取所有图片的路径及其对应的类别
#         self.image_paths = []
#         self.labels = []
#         # import pdb
#         # pdb.set_trace()
#         for class_name in self.classes:
#             class_dir = os.path.join(self.root_dir, class_name)
#             if os.path.isdir(class_dir):
#                 for img_file in os.listdir(class_dir):
#                     if img_file.endswith(('.jpg', '.JPEG', '.png')):  # 支持的图片格式
#                         self.image_paths.append(os.path.join(class_dir, img_file))
#                         self.labels.append(self.class_to_idx[class_name])

#         self.num_classes = len(self.classes)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # 获取图像路径和标签
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]

#         # 打开图像
#         image = Image.open(img_path).convert("RGB")

#         # 如果该图像的类别是需要遗忘的类别，则随机替换标签
#         if label == self.class_to_forget:
#             label = np.random.randint(0, self.num_classes)

#         # 如果有图像变换操作，则进行变换
#         if self.transform:
#             image = self.transform(image)

#         return image, label, img_path



class ImagenetteCustom(Dataset):
    def __init__(self, root_dir='/data2/ljq/do_train/stable-diffusion/imagenette', split='train', class_to_forget=11, transform=None, target_mask_size=(40, 40)):
        """
        root_dir: 数据集根目录
        split: 'train' 或 'val'，决定加载训练集或验证集
        class_to_forget: 如果指定了这个类别，标签会被随机替换或处理
        transform: 数据变换操作
        target_mask_size: 固定的 mask 尺寸，默认为下采样后的尺寸 (宽, 高)
        """
        self.root_dir = os.path.join(root_dir, split)
        self.json_dir = os.path.join(root_dir, 'Hello_Kitty_json')  # 假设 json 文件在此文件夹中
        self.class_to_forget = class_to_forget
        self.transform = transform
        self.target_mask_size = target_mask_size  # 设定固定的 mask 尺寸

        # 获取类名（文件夹名）及对应的标签
        self.classes = sorted(os.listdir(self.root_dir))  # 文件夹名字即类别名
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 获取所有图片的路径及其对应的类别，只保留需要遗忘的类
        self.image_paths = []
        self.labels = []
        self.json_paths = []  # 对应的 JSON 文件路径

        # 如果指定了 class_to_forget，则进行过滤
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[class_name]

                # 仅保留需要遗忘的类的数据
                if self.class_to_forget is not None and class_idx != self.class_to_forget:
                    continue  # 跳过不需要遗忘的类
                # pdb.set_trace()
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.jpg', '.JPEG', '.png')):  # 支持的图片格式
                        self.image_paths.append(os.path.join(class_dir, img_file))
                        self.labels.append(class_idx)

                        # 检查是否有对应的json文件
                        json_file = img_file.split('.')[0] + '.json'
                        json_file_path = os.path.join(self.json_dir, json_file)

                        # 如果 json 文件存在，添加路径；否则，添加空路径
                        if os.path.exists(json_file_path):
                            self.json_paths.append(json_file_path)
                        else:
                            self.json_paths.append("")  # 添加空路径，避免报错

        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.image_paths)

    def get_segmentation_mask(self, json_path, img_size):
        """
        根据JSON标注文件生成 segmentation mask
        """
        if json_path == "":  # 如果json文件为空，直接返回全零mask
            return Image.new('L', img_size, 0)  # 全零的mask

        with open(json_path, 'r') as f:
            data = json.load(f)

        img_w, img_h = img_size
        mask = Image.new('L', (img_w, img_h), 0)  # 创建一个空的mask

        # Draw polygons based on points in JSON file
        for shape in data['shapes']:
            points = [(point[0], point[1]) for point in shape['points']]
            ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)

        return mask  # 返回 PIL Image 格式的 mask



    # def get_segmentation_mask(self, json_path, img_size, dilation_size=5):
    #     """
    #     根据JSON标注文件生成 segmentation mask，并随机扩展一定的像素（dilation_size）。
    #     """
    #     if json_path == "":  # 如果json文件为空，直接返回全零mask
    #         return Image.new('L', img_size, 0)  # 全零的mask

    #     with open(json_path, 'r') as f:
    #         data = json.load(f)

    #     img_w, img_h = img_size
    #     mask = Image.new('L', (img_w, img_h), 0)  # 创建一个空的mask

    #     # Draw polygons based on points in JSON file
    #     for shape in data['shapes']:
    #         points = [(point[0], point[1]) for point in shape['points']]
    #         ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)

    #     # 将mask转换为numpy格式
    #     mask = np.array(mask)

    #     # 执行膨胀操作，随机扩大 mask 的区域
    #     kernel = np.ones((dilation_size, dilation_size), np.uint8)
    #     mask_dilated = cv2.dilate(mask, kernel, iterations=1)  # 使用OpenCV进行膨胀操作

    #     return Image.fromarray(mask_dilated)  # 返回膨胀后的mask
    









    def __getitem__(self, idx):
        # 获取图像路径和标签
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        json_path = self.json_paths[idx]

        # 打开图像
        image = Image.open(img_path).convert("RGB")

        # 获取图像的原始尺寸
        img_size = image.size  # 原图像尺寸

        # 根据JSON生成 segmentation mask
        mask = self.get_segmentation_mask(json_path, img_size)
        # print(np.array(mask).sum())
        # 只对mask进行resize到下采样尺寸（如下采样8倍后）
        mask = mask.resize(self.target_mask_size, Image.NEAREST)  # 使用 NEAREST 保持mask的二值性
        mask = np.array(mask)  # 将 mask 转为 numpy 格式
        # print(mask.sum())
        # 如果有图像变换操作，则对图像进行变换
        if self.transform:
            image = self.transform(image)

        # 返回 image, label 和 resize 后的 mask
        return image, label, mask


class NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image


class NOT_NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/not-nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image


def setup_model(config, ckpt, device):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def setup_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", class_to_forget, transform)
    # train_set = Imagenette('train', transform)

    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_ga_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {class_to_name[label]}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] == class_to_forget]

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


# def setup_remain_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
#     interpolation = INTERPOLATIONS[interpolation]
#     transform = get_transform(interpolation, image_size)

#     train_set = Imagenette("train", transform=transform)
#     descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
#     filtered_data = [data for data in train_set if data[1] != class_to_forget]

#     train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
#     return train_dl, descriptions


# def setup_forget_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
#     interpolation = INTERPOLATIONS[interpolation]
#     transform = get_transform(interpolation, image_size)

#     train_set = Imagenette("train", transform=transform)
#     import pdb
#     pdb.set_trace()
#     descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
#     filtered_data = [data for data in train_set if data[1] == class_to_forget]
#     train_dl = DataLoader(filtered_data, batch_size=batch_size)
#     return train_dl, descriptions



# def setup_remain_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
#     interpolation = INTERPOLATIONS[interpolation]
#     transform = get_transform(interpolation, image_size)

#     train_set = ImagenetteCustom(split='train', transform=transform)
#     pdb.set_trace()
#     descriptions = [f"an image of a {class_to_name[label]}" for label in train_set.class_to_idx.keys()]
#     filtered_data = [data for data in train_set if data[1] != class_to_forget]

#     train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
#     return train_dl, descriptions


# def setup_forget_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
#     interpolation = INTERPOLATIONS[interpolation]
#     transform = get_transform(interpolation, image_size)

#     train_set = ImagenetteCustom(split='train', transform=transform)
#     # import pdb
#     # pdb.set_trace()
#     descriptions = [f"an image of a {class_to_name[label]}" for label in train_set.class_to_idx.keys()]
#     filtered_data = [data for data in train_set if data[1] == class_to_forget]
#     train_dl = DataLoader(filtered_data, batch_size=batch_size)
#     return train_dl, descriptions

def setup_remain_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = ImagenetteCustom(split='train', transform=transform)
    # pdb.set_trace()
    descriptions = [f"an image of a {class_to_name[label]}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] != class_to_forget]

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_forget_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = ImagenetteCustom(split='train', transform=transform)
    # import pdb
    # pdb.set_trace()
    descriptions = [f"an image of a {class_to_name[label]}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] == class_to_forget]
    train_dl = DataLoader(train_set, batch_size=batch_size)
    return train_dl, descriptions



def setup_forget_nsfw_data(batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = NSFW(transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    remain_set = NOT_NSFW(transform=transform)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl
