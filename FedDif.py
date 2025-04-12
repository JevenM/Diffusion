# -*- coding:utf-8 -*-
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
import numpy as np
import copy
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import logging
from datetime import datetime
import subprocess as sp
from copy import deepcopy
import math
import random
from contextlib import contextmanager
from torch import optim, nn
from torchvision import datasets, transforms
from sklearn.utils import shuffle

"""
FL训练一个diffusion全局模型
数据集: mnist、Cifar10

参考: Federated Learning with Diffusion Models for Privacy-Sensitive Vision Tasks

run: pymao FedDif.py
"""

# 设置cuda
torch.backends.cudnn.benchmark = True  # 加速固定输入
torch.backends.cudnn.deterministic = True  # 确定性算法
# 选择GPU
device_id = 1
# 1:iid 0:non-iid 2:direchlet dist
is_iid = 0
num_users = 10  # 10
# data_dir = '/data/mwj/mycode1/FGL_FrameWork/data' #'Mnist' #'SVHN' #'Cifar10' #'stl10'
# data_dir = './data'
# use_dataset = 'SVHN'
data_dir = "./Brain-Tumor-Classification-DataSet-master"
use_dataset = "MRI"
inChannel = 3
verbose = True
local_train_bs = 256
test_bs = 256
# fedavg or fedatt?
use_avg = False
# R: 300
num_rounds = 300
num_classes = 4  # 10
# E
local_epoch = 10
frac_users = 1  # 1 0.4 0.7 1.0
train_ratio = 0.8
learning_rate = 2e-4
# dirichlet param: 0.1,0.5,5
DALPHA = 0.5
# The number of timesteps to use when sampling
steps = 300
# The amount of noise to add each timestep when sampling
# 0 = no noise (DDIM)
# 1 = full noise (DDPM)
eta = 1.0
ema_decay = 0.998


de_str = f"cuda:{device_id}"
device = torch.device(de_str if torch.cuda.is_available() else "cpu")

# 报错详细信息
torch.autograd.set_detect_anomaly(True)

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 是否使用多GPU
USE_MULTI_GPU = False
# 检测机器是否有多张显卡
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
else:
    MULTI_GPU = False
    # 参数
    if not torch.cuda.is_available():
        device_name = "cpu"
        device = torch.device("cpu")
    else:
        # 设备选择：60002 上 2 号卡, jupyter方式
        # h = !hostname
        # device = torch.device('cuda:2' if h[0] =='60002' else de_str)
        # py代码文件方式，好像在tmux中不行，应该是60002，现在是master了
        h = sp.getoutput("uname -n")
        print(h)
        device = torch.device("cuda:2" if h == "60002" else de_str)

print(f"use device: {device}")

# if torch.cuda.is_available():
#     print(torch.cuda.get_device_name())


if is_iid == 1:
    iid_str = f"iid"
elif is_iid == 0:
    iid_str = f"iid{is_iid}"
else:
    iid_str = f"iid{is_iid}-da{DALPHA}"
comments = f"dif-{iid_str}-avg{use_avg}-{use_dataset}-U{num_users}-F{frac_users}-lrfc{learning_rate}-E{local_epoch}-R{num_rounds}-BS{local_train_bs}-tr{train_ratio}_s{steps}"
print(comments)

result_name = (
    str(datetime.now())
    .split(".")[0]
    .replace(" ", "_")
    .replace(":", "_")
    .replace("-", "_")
    + "_"
    + comments
)

# curr_working_dir = os.getcwd()
save_dir = os.path.join("./results", result_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, "gen_images")
log_name = os.path.join(save_dir, "train.log")


def set_logger(log_file_path="", file_name=""):

    # 创建一个 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个处理器，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 设置控制台日志级别

    # 创建一个处理器，用于将日志输出到文件

    file_handler = logging.FileHandler(log_file_path + file_name)
    file_handler.setLevel(logging.DEBUG)  # 设置文件日志级别

    # 创建一个格式化器，用于设置日志的格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 示例日志
    # logger.debug('This is a debug message')
    # logger.info('This is an info message')
    # logger.warning('This is a warning message')
    # logger.error('This is an error message')
    # logger.critical('This is a critical message')
    return logger


log = set_logger(file_name=log_name)

# curr_dir = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")
curr_dir = result_name
runs_dir = "./runs/"

curr_dir = runs_dir + curr_dir
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)

summary_writer = SummaryWriter(log_dir=curr_dir, comment=comments)

log.info(result_name)

# ------------ Define the model (a residual U-Net) -----------------------


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            [
                nn.Conv2d(c_in, c_mid, 3, padding=1),
                nn.Dropout2d(0.1, inplace=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_out, 3, padding=1),
                nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
                nn.ReLU(inplace=True),
            ],
            skip,
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        # print(f"input: {input.shape}")
        # print(f"main: {self.main(input).shape}")
        # print(f"skip: {self.skip(input).shape}")
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class Diffusion(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(  # 32x32
            ResConvBlock(in_channel + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock(
                [
                    nn.AvgPool2d(2),  # 32x32 -> 16x16
                    ResConvBlock(c, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock(
                        [
                            nn.AvgPool2d(2),  # 16x16 -> 8x8
                            ResConvBlock(c * 2, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            SkipBlock(
                                [
                                    nn.AvgPool2d(2),  # 8x8 -> 4x4
                                    ResConvBlock(c * 4, c * 8, c * 8),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    ResConvBlock(c * 8, c * 8, c * 4),
                                    nn.Upsample(scale_factor=2),
                                ]
                            ),  # 4x4 -> 8x8
                            ResConvBlock(c * 8, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 2),
                            nn.Upsample(scale_factor=2),
                        ]
                    ),  # 8x8 -> 16x16
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c),
                    nn.Upsample(scale_factor=2),
                ]
            ),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, in_channel, dropout_last=False),
        )

    def forward(self, input, log_snrs, cond):
        # print(f"input shape {input.shape}")
        timestep_embed = expand_to_planes(
            self.timestep_embed(log_snrs[:, None]), input.shape
        )
        class_embed = expand_to_planes(self.class_embed(cond), input.shape)
        return self.net(torch.cat([input, class_embed, timestep_embed], dim=1))


# --------------------- Utilities ------------------


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the noise schedule and sampling loop


def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given the log SNR for a timestep."""
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.special.expm1(1e-4 + 10 * t**2).log()


class ClientModel(nn.Module):
    def __init__(self, in_channel=3, device=torch.device("cuda")):
        super(ClientModel, self).__init__()
        self.device = device
        self.model = Diffusion(in_channel).to(device)

    def forward(self, input, log_snrs, cond):
        y = self.model(input, log_snrs, cond)
        return y


def dirichlet_split_noniid(train_labels, alpha_, n_clients):
    """
    Dirichlet distribution with `alpha_` parameter divides the data index into `n_clients` subset

    Args:
    ---
        `train_labels`: List of class labels for raw dataset samples
        `alpha_`: parameter of dirichlet distribution
        `n_clients`: num of users
    From:
    ---
        https://zhuanlan.zhihu.com/p/468992765

    Example:
    ---
    Parameters:
        alpha_: sequence of floats, length k alpha in distribution function with a length of k
        size: int or tuple of ints, optional output shape, the last dimension is k, given the shape (mxn),
        the return mxn random vector which length is k

    Return:
        ndarray: the shape ids (size, k)

    Code:
        >>> s = np.random.dirichlet((10, 5, 3), size=(2, 2))
        >>> print(s)
        >>> [
                [
                    [0.82327647 0.09820451 0.07851902]
                    [0.50861077 0.4503409  0.04104833]
                ][
                    [0.31843167 0.22436547 0.45720285]
                    [0.40981943 0.40349597 0.1866846 ]
                ]
            ]
        >>> shape: 2*2*3
    """
    if isinstance(train_labels, list):
        n_classes = max(train_labels) + 1
    else:
        n_classes = train_labels.max() + 1
    assert n_classes == num_classes
    n_classes = num_classes
    label_distribution = np.random.dirichlet([alpha_] * n_clients, n_classes)
    # The category label distribution matrix X of (k, n) records the number of each category occupied by each client

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    # Record the sample subscript corresponding to each k categories

    client_idcs = [[] for _ in range(n_clients)]
    # Record the index of the sample set corresponding to n clients respectively
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split divides samples of category K into n subsets in proportion
        # for i, idcs are the index of the sample set corresponding to traversing the ith client
        for i, idcs in enumerate(
            np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
        ):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    dict_users_ = {i: np.array([]) for i in range(n_clients)}
    for idx, arr in enumerate(client_idcs):
        dict_users_[idx] = arr
    return dict_users_


def dataset_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    # 100: args.shots
    # num_items = 100
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def dataset_noniid_ori(dataset, num_users, labels):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :return: userid: 500 samples' index
    """
    # 40,000 training imgs -->  2000 imgs/shard X 20 shards
    len_datasets = len(dataset)
    num_shards = num_users * 2
    num_imgs = len_datasets // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    data_len = num_shards * num_imgs
    idxs = np.arange(data_len, dtype=np.int32)
    labels = labels[:data_len]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        # 随机选择 0-200 中的两个数
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # 删除对应的 rand_set（两个数）
        idx_shard = list(set(idx_shard) - rand_set)
        # 将500个数分为两次给
        for rand in rand_set:
            # 从rand索引开始每次取 num_imgs 个数分给一个客户端， rand最大199
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )

    return dict_users


# -------------------------------- dataset --------------------------------------
# 自定义数据集类
class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


if use_dataset == "Cifar10":
    # 数据加载和预处理
    trans_cifar10_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47375134, 0.47303376, 0.42989072],
                std=[0.25467148, 0.25240466, 0.26900575],
            ),
        ]
    )

    trans_cifar10_train = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48836562, 0.48134598, 0.4451678],
                std=[0.24833508, 0.24547848, 0.26617324],
            ),
        ]
    )

    # 下载数据集
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, transform=trans_cifar10_train, download=False
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, transform=trans_cifar10_test, download=False
    )
elif use_dataset == "Mnist":
    inChannel = 1
    # 数据加载和预处理
    trans_mnist_train = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trans_mnist_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # 下载数据集
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, transform=trans_mnist_train, download=True
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, transform=trans_mnist_test, download=True
    )
elif use_dataset == "SVHN":
    # 数据加载和预处理
    trans_svhn = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.106], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = datasets.SVHN(
        root=data_dir, download=True, split="train", transform=trans_svhn
    )

    test_dataset = datasets.SVHN(
        root=data_dir, download=True, transform=trans_svhn, split="test"
    )
elif use_dataset == "stl10":
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # Convert PIL image to PyTorch tensor
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize the data
        ]
    )
    # Load the STL-10 dataset
    train_dataset = datasets.STL10(
        root=data_dir, split="train", download=True, transform=transform
    )
    test_dataset = datasets.STL10(
        root=data_dir, split="test", download=True, transform=transform
    )
elif use_dataset == "MRI":
    image_size = 32
    # 标签和图片大小
    labels_txt = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    # 图像预处理
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    # 加载数据
    X_data = []
    Y_data = []

    for phase in ["Training", "Testing"]:
        for idx, label in enumerate(labels_txt):
            folder = os.path.join(
                "Brain-Tumor-Classification-DataSet-master", phase, label
            )
            for img_name in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, img_name))
                img = cv2.resize(img, (image_size, image_size))
                X_data.append(img)
                Y_data.append(idx)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    X_data, Y_data = shuffle(X_data, Y_data, random_state=101)

    # 划分训练和测试集
    split_ratio = 0.8
    total_samples = len(X_data)
    train_size = int(split_ratio * total_samples)
    test_size = total_samples - train_size

    X_train, X_test = X_data[:train_size], X_data[train_size:]
    y_train, y_test = Y_data[:train_size], Y_data[train_size:]

    train_dataset = BrainTumorDataset(X_train, y_train, transform)
    test_dataset = BrainTumorDataset(X_test, y_test, transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_bs,
    shuffle=False,
    num_workers=8,
    drop_last=False,
    pin_memory=True,
)

from torch.utils.data import random_split

train_d_len = len(train_dataset)
generator = torch.Generator().manual_seed(seed)
dataset_train, dataset_valid = random_split(
    train_dataset,
    [int(train_ratio * train_d_len), train_d_len - int(train_ratio * train_d_len)],
    generator,
)
# labels_ = np.array(train_dataset.targets)
# 因为采样了一部分，所以通过遍历subset_dataset来获取每个样本的标签，注意加上np.array(
subset_labels = np.array([dataset_train[i][1] for i in range(len(dataset_train))])

if is_iid == 1:
    # 训练和测试时客户端id和数据id的字典
    dict_users = dataset_iid(dataset_train, num_users)
elif is_iid == 0:
    dict_users = dataset_noniid_ori(dataset_train, num_users, subset_labels)
elif is_iid == 2:
    # dirichlet
    dict_users = dirichlet_split_noniid(subset_labels, DALPHA, num_users)
dict_users_lt = dataset_iid(dataset_valid, num_users)

log.info(f"user ids: {dict_users.keys()}")

total_samples = 0
for u in range(num_users):
    total_samples += len(dict_users[u])
log.info(f"total_samples: {total_samples}")

# build model
global_model = None
global_model = ClientModel(inChannel, device=device)
global_model_ema = deepcopy(global_model)
log.info(global_model)
log.info(
    "Diffusion model total params: %.2fM"
    % (sum(p.numel() for p in global_model.parameters()) / 1000000.0)
)
# 打印模型的参数列表
for name, param in global_model.named_parameters():
    log.info(
        f"global model parameter name: {name}, Shape: {param.shape}, grad: {param.requires_grad}"
    )


# Use a low discrepancy quasi-random sequence to sample uniformly distributed
# timesteps. This considerably reduces the between-batch variance of the loss.
rng = torch.quasirandom.SobolEngine(1, scramble=True)


def eval_loss(model, rng, reals, classes):
    # Draw uniformly distributed continuous timesteps
    t = rng.draw(reals.shape[0])[:, 0].to(device)

    # Calculate the noise schedule parameters for those timesteps
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)
    weights = log_snrs.exp() / log_snrs.exp().add(1)

    # Combine the ground truth images and the noise
    alphas = alphas[:, None, None, None]
    sigmas = sigmas[:, None, None, None]
    noise = torch.randn_like(reals)
    noised_reals = reals * alphas + noise * sigmas
    targets = noise * alphas - reals * sigmas

    # Compute the model output and the loss.
    with torch.cuda.amp.autocast():
        v = model(noised_reals, log_snrs, classes)
        return (v - targets).pow(2).mean([1, 2, 3]).mul(weights).mean()


@torch.no_grad()
def sample(model, x, steps, eta, classes):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in range(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            # print(f"x shape: {x.shape}")
            v = model(x, ts * log_snrs[i], classes).float()
        # print(f"v shape: {v.shape}")
        # print(f"x1 shape: {x.shape}")
        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]
        # print(f"x2 shape: {x.shape}")
        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = (
                eta
                * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt()
                * (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            )
            adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma
            # print(f"x3 shape: {x.shape}")

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma
        # print(f"@@@@@@@@@x.shape:{x.shape}")

    # If we are on the last timestep, output the denoised image
    return pred


@torch.no_grad()
@torch.random.fork_rng(devices=[device_id])
def demo(model, c_id, gen_img_num=100, g_name="None", in_channel=3):
    eval_mode(model)
    torch.manual_seed(seed)
    # print(f"in_channel: {in_channel}")
    noise = torch.randn([gen_img_num, in_channel, 32, 32], device=device)
    num_of_each_class = gen_img_num // num_classes
    # 生成样本的类别，每个元素重复10次，意思是生成10个类别，每个类别10个样本
    fakes_classes = torch.arange(num_classes, device=device).repeat_interleave(
        num_of_each_class, 0
    )
    # print(f"noise shape: {noise.shape}")
    fakes = sample(model, noise, steps, eta, fakes_classes)
    if c_id is not None:
        whole_img_dir = save_path + "/c_" + str(c_id)
    elif g_name != "None":
        whole_img_dir = save_path + "/" + g_name
    os.makedirs(whole_img_dir, exist_ok=True)
    # 存成整体图片
    utils.save_image(
        fakes,
        whole_img_dir + f"/whole_{gen_img_num}.png",
        nrow=num_of_each_class,
        normalize=True,
        scale_each=True,
    )
    # 存成单个图像
    for i, img in enumerate(fakes):
        class_dir = whole_img_dir + "/" + str(i // num_of_each_class)
        os.makedirs(class_dir, exist_ok=True)
        # print(img.shape)
        utils.save_image(
            img,
            class_dir + f"/demo_{i%num_of_each_class}.png",
            nrow=1,
            normalize=True,
            scale_each=True,
        )


# 定义客户端本地更新训练的类
class LocalUpdate(object):
    def __init__(
        self,
        id,
        dataset=None,
        device=device,
        idxs=None,
        local_model=None,
        local_bs=local_train_bs,
        local_epochs=local_epoch,
    ):
        """
        Args:
        ---
            `dataset`: train dataset
            `idxs`: samples index of a client
        """
        self.device = device
        self.local_bs = local_bs
        self.local_epochs = local_epochs
        self.local_model = local_model
        self.id = id
        self.trainloader = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=self.local_bs,
            shuffle=True,
            pin_memory=True,
            drop_last=False
            # , sampler=train_sampler
            ,
            num_workers=2,
        )
        self.train_samples_num = len(self.trainloader.dataset)
        self.opt = optim.Adam(self.local_model.parameters(), lr=learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()
        self.model_ema = deepcopy(self.local_model)

    def train(self):
        train_epoch_loss1 = []

        # self.args.local_epochs: the number of local epochs
        for iter in range(self.local_epochs):
            self.local_model.train()
            tloss = 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.opt.zero_grad()
                # Evaluate the loss
                loss = eval_loss(self.local_model, rng, images, labels)
                tloss += loss.item()
                # Do the optimizer step and EMA update
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                ema_update(
                    self.local_model, self.model_ema, 0.95 if iter < 20 else ema_decay
                )
                self.scaler.update()

                if verbose and batch_idx % 10 == 0:
                    log.info(
                        "Update Epoch-train: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                            iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss,
                        )
                    )

            train_epoch_loss1.append(tloss / len(self.trainloader))
            if (iter + 1) % self.local_epochs == 0:
                val_ls = self.val()
                log.info(f"Validation: local epoch: {iter}, loss: {val_ls:g}")
                demo(self.model_ema, self.id, in_channel=inChannel)
            self.save()
            # 要不要加？
            # del images,labels,batch_idx
            # gc.collect()  #清除数据与变量相关的缓存
            # torch.cuda.empty_cache()

        return (
            self.local_model.model.state_dict(),
            self.model_ema.model.state_dict(),
            sum(train_epoch_loss1) / len(train_epoch_loss1),
        )

    @torch.no_grad()
    @torch.random.fork_rng(devices=[device_id])
    def val(self):
        eval_mode(self.model_ema)
        torch.manual_seed(seed)
        rng = torch.quasirandom.SobolEngine(1, scramble=True)
        total_loss = 0
        count = 0
        for i, (reals, classes) in enumerate(testloader_list[self.id]):
            reals = reals.to(device)
            classes = classes.to(device)

            loss = eval_loss(self.model_ema, rng, reals, classes)

            total_loss += loss.item() * len(reals)
            count += len(reals)
        loss = total_loss / count
        return loss

    def save(self):
        save_dir_ = os.path.join("/home/mwj/mycode2/Diffusion/models", result_name)
        if not os.path.exists(save_dir_):
            os.makedirs(save_dir_)
        filename = f"{save_dir_}/{use_dataset}_diffusion_c{self.id}.pth"
        obj = {
            "model": self.local_model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(obj, filename)


# 数据集分割
class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.

    For supervised learning of fedavg.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        # self.idxs = list(idxs)
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
        # return image, torch.tensor(label)


@torch.no_grad()
@torch.random.fork_rng(devices=[device_id])
# 对所有客户端模型评估
def test_inference_lt_prompt():
    """
    Returns the test accuracy and loss of all local models.
    lt: local test
    """

    loss_list = []
    torch.manual_seed(seed)
    rng = torch.quasirandom.SobolEngine(1, scramble=True)

    for idx in range(num_users):
        model_ema = local_update_list[idx].model_ema
        eval_mode(model_ema)
        total_loss = 0
        loss = 0
        val_total = 0
        model_ema.to(device)
        model_ema.eval()
        with torch.no_grad():
            for batch_idx, (val_inputs, val_labels) in enumerate(testloader_list[idx]):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                loss = eval_loss(model_ema, rng, val_inputs, val_labels)
                total_loss += loss.item() * len(val_inputs)
                val_total += val_inputs.size(0)

        # mean loss of batches
        log.info(
            "test_inference_lt_prompt：| Client: {} | Local Test Loss: {:.5f}".format(
                idx, total_loss / val_total
            )
        )
        loss_list.append(total_loss / val_total)

    return loss_list


# 修改评估单个模型的函数
@torch.no_grad()
@torch.random.fork_rng(devices=[device_id])
def inference_client(model_ema, idx):
    """
    Returns the inference accuracy and loss.
    """
    eval_mode(model_ema)
    model_ema.to(device)
    model_ema.eval()

    total_loss, val_total = 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (val_inputs, labels) in enumerate(testloader_list[idx]):
            val_inputs, labels = val_inputs.to(device), labels.to(device)
            loss = eval_loss(model_ema, rng, val_inputs, labels)
            val_total += val_inputs.size(0)
            total_loss += loss.item() * val_inputs.size(0)

        mean_loss = total_loss / val_total
        log.info(
            "inference_client: | Client: {} | Local Test Loss: {:.5f}".format(
                idx, mean_loss
            )
        )

    return mean_loss


# 修改评估模型的函数
@torch.no_grad()
@torch.random.fork_rng(devices=[device_id])
def test_global(model_ema, testloader, round):
    """
    Returns the global test accuracy and loss.
    """
    eval_mode(model_ema)
    model_ema.to(device)
    model_ema.eval()
    total_loss, val_total = 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (val_inputs, labels) in enumerate(testloader):
            val_inputs, labels = val_inputs.to(device), labels.to(device)
            loss = eval_loss(model_ema, rng, val_inputs, labels)
            val_total += val_inputs.size(0)
            total_loss += loss.item() * val_inputs.size(0)

        mean_loss = total_loss / val_total
        log.info(f"Round {round}, Global Test Loss: {mean_loss}")

    return mean_loss


def save(model, epoch):
    save_dir_ = os.path.join("/home/mwj/mycode2/Diffusion/models", result_name)
    if not os.path.exists(save_dir_):
        os.makedirs(save_dir_)
    filename = f"{save_dir_}/{use_dataset}_global_diffusion_e{epoch}.pth"
    obj = {
        "model": model.state_dict(),
        "epoch": epoch,
    }
    torch.save(obj, filename)


# 定义联邦平均，w是权重参数
def FedAvg(w):
    """
    FedAvg aggregation algorithm. TODO: not weighted averaging?

    Returns the average of the weights.
    """
    assert len(w) > 0 and w[0] is not None, "list should be greater than 0"
    # print("weights length:", len(w))
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        # print("k={0}, w_avg.keys()= {1}".format(k, w_avg.keys()))
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def aggregate_att(w_clients, w_server):
    import copy
    import torch
    import torch.nn.functional as F

    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    avg_g_w = FedAvg(w_clients)
    global_next = copy.deepcopy(w_server)
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm(
                w_server[k].cpu() - w_clients[i][k].cpu(), p=2
            ).item()
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1 / torch.std(att[k]))
        att[k] = F.softmax(att[k] / std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, 1)
    for k in w_next.keys():
        global_next[k] = (0.5) * w_next[k] + 0.5 * avg_g_w[k]
    return global_next


# 客户端的测试数据加载器
testloader_list = []
for idx in range(num_users):
    testloader_list.append(
        DataLoader(
            DatasetSplit(dataset_valid, dict_users_lt[idx]),
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )
    )

# 本地更新类的对象列表
local_update_list = [
    LocalUpdate(
        id=idx,
        dataset=dataset_train,
        device=device,
        idxs=dict_users[idx],
        local_model=ClientModel(inChannel, device=device),
    )
    for idx in range(num_users)
]

# train
loss_train_list = []
# val
loss_test_list = []
# total users id
idxs_users = np.arange(num_users, dtype=np.int32)

global_list_ls = []
global_ema_list_ls = []

# 获取当前时间
start_time = datetime.now()

# phase1: local training
log.info("phase1: users local training.")
with trange(num_rounds, colour="red", desc="Round") as t:
    for round in t:
        # selected clients
        idxs_users = np.random.choice(
            np.arange(num_users, dtype=np.int32),
            int(num_users * frac_users),
            replace=False,
        )
        log.info(f"selected users index:{idxs_users}")
        # clear loss and weight list each round
        local_losses = []
        local_weights = []
        local_ema_weights = []
        # training on selected clients
        with trange(len(idxs_users), colour="green", leave=False) as users:
            for i in users:
                idx = idxs_users[i]
                w_l, w_ema_l, loss_l = local_update_list[idx].train()
                local_weights.append(w_l)
                local_ema_weights.append(w_ema_l)
                local_losses.append(loss_l)
                summary_writer.add_scalar(
                    "Train-Loss/user" + str(idx), loss_l, round + 1
                )
                users.set_description("C:{:d}".format(idx))

        loss_avg = sum(local_losses) / len(local_losses)
        loss_train_list.append(loss_avg)

        log.info("test before replacing parameters")
        # before replacing parameters evaluate
        loss_list_b = test_inference_lt_prompt()
        mean_ls_b = np.mean(loss_list_b, dtype=np.float64)
        summary_writer.add_scalar("Test-Loss/alluser-b", mean_ls_b, round + 1)

        if use_avg:
            global_weights = FedAvg(local_weights)
            global_ema_weights = FedAvg(local_ema_weights)
        else:
            global_weights = aggregate_att(
                local_weights, global_model.model.state_dict()
            )
            global_ema_weights = aggregate_att(
                local_ema_weights, global_model_ema.model.state_dict()
            )

        # distribute to all clients to update local models
        # TODO 这里注意有修改，原本是idxs_users，这就意味着分发的时候不是全部客户端
        for idx in range(num_users):
            local_update_list[idx].local_model.model.load_state_dict(global_weights)
            # local_update_list[idx].model_ema.model.load_state_dict(global_ema_weights)

        if verbose:
            log.info(
                "Global Round {:3d}, Train average loss {:.5f}".format(
                    round + 1, loss_avg
                )
            )

        log.info("local model evaluate process.")

        # type-1 evaluate
        loss_list = test_inference_lt_prompt()
        mean_ls = np.mean(loss_list, dtype=np.float64)

        # type-3 evaluate
        # Calculate avg test accuracy over all users at every epoch
        list_loss = []
        for idx in range(num_users):
            # visual_model, idx, weight_decay=0.1
            loss = inference_client(model_ema=local_update_list[idx].model_ema, idx=idx)
            list_loss.append(loss)

        mean_ls3 = sum(list_loss) / len(list_loss)

        summary_writer.add_scalar("Test-Loss/alluser3", mean_ls3, round + 1)
        summary_writer.add_scalar("Test-Loss/alluser", mean_ls, round + 1)

        loss_test_list.append(mean_ls)
        log.info(
            "For all users, mean of test loss is {:.5f}, std of test loss is {:.5f}".format(
                mean_ls, np.std(loss_list, dtype=np.float64)
            )
        )

        if round == num_rounds - 1:
            try:
                dif_w_path = save_dir + "/diffusion_r" + str(round + 1) + ".pth"
                dif_w_ema_path = save_dir + "/diffusion_ema_r" + str(round + 1) + ".pth"
                torch.save(global_weights, dif_w_path)
                torch.save(global_ema_weights, dif_w_ema_path)
            except:
                pass

        global_model.model.load_state_dict(global_weights)
        global_model_ema.model.load_state_dict(global_ema_weights)
        # test global get its loss, 评估时应该看中ema模型的性能
        g_m_ls = test_global(global_model, test_loader, round)
        g_emam_ls = test_global(global_model_ema, test_loader, round)
        # get global sample images, 评估时应该看中ema模型的性能
        demo(global_model, c_id=None, g_name="global", in_channel=inChannel)
        demo(global_model_ema, c_id=None, g_name="global_ema", in_channel=inChannel)
        global_list_ls.append(g_m_ls)
        global_ema_list_ls.append(g_emam_ls)
        summary_writer.add_scalar("Test-Loss/global", g_m_ls, round + 1)
        summary_writer.add_scalar("Test-Loss/global_ema", g_emam_ls, round + 1)
        # Postfix will be displayed on the right,
        # formatted automatically based on argument's datatype
        t.set_postfix(tavgls="{:.3f}".format(loss_avg))
#         torch.cuda.empty_cache()

# 最后保存一次
save(global_model, num_rounds)
save(global_model_ema, num_rounds)

end_time = datetime.now()
h_, remainder_ = divmod((end_time - start_time).seconds, 3600)
m_, s_ = divmod(remainder_, 60)
time_str_ = "Time %02d:%02d:%02d" % (h_, m_, s_)
log.info(f"\n Total Run {time_str_}")

log.info(f"loss_train_list: {loss_train_list}")
log.info(f"loss_test_list: {loss_test_list}")
log.info(f"global_list_ls: {global_list_ls}")
log.info(f"global_ema_list_ls: {global_ema_list_ls}")

import matplotlib.pyplot as plt

# 生成 x 轴数据，假设是 epoch 数
epochs = list(range(1, len(loss_train_list) + 1))

# 画曲线图
plt.plot(epochs, loss_train_list, label="Mean Train Loss")
plt.plot(epochs, loss_test_list, label="Mean Test Loss")
plt.plot(epochs, global_list_ls, label="Global Loss")
plt.plot(epochs, global_ema_list_ls, label="Global ema Loss")

# 添加标题和标签
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# 添加图例
plt.legend()

# 保存为文件（可以选择不同的格式，如PNG、PDF等）
plt.savefig(save_dir + "/loss_curves.png", dpi=300, bbox_inches="tight")
log.info(result_name)
