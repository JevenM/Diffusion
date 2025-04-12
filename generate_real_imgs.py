import random
import cv2
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import os
import torchvision
import torchvision.utils as vutils


class BrainTumorDataset(data_utils.Dataset):
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


# 数据集分割
class DatasetSplit(data_utils.Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.

    For supervised learning of fedavg.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
        # return image, torch.tensor(label)


num_clients = 10
datasets = ["MRI"]
num_classes = 4
num_of_per_class = 10
total_num = num_classes * num_of_per_class


def get_real_data():
    for ds in datasets:
        data_dir = "Brain-Tumor-Classification-DataSet-master"
        if ds == "MRI":
            image_size = 32
            # 标签和图片大小
            labels_txt = [
                "glioma_tumor",
                "meningioma_tumor",
                "no_tumor",
                "pituitary_tumor",
            ]
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

            for idx, label in enumerate(labels_txt):
                folder = os.path.join(
                    "Brain-Tumor-Classification-DataSet-master", "Training", label
                )
                for img_name in os.listdir(folder):
                    img = cv2.imread(os.path.join(folder, img_name))
                    img = cv2.resize(img, (image_size, image_size))
                    X_data.append(img)
                    Y_data.append(idx)

            X_data = np.array(X_data)
            Y_data = np.array(Y_data)

            train_dataset = BrainTumorDataset(X_data, Y_data, transform)
            # test_dataset = BrainTumorDataset(X_test, y_test, transform)
        # 手动划分一下真实和虚假的图像,40个,每个类别10张
        indices_all = []
        # 遍历每个类别
        for class_idx in range(num_classes):
            # 筛选当前类别的前40个数据样本
            indices = [
                i for i, label in enumerate(train_dataset.labels) if label == class_idx
            ][:num_of_per_class]

            # # 遍历选定的样本
            # for i, idx in enumerate(indices):
            #     image, label = dataset_[idx]
            #     image = transforms.ToPILImage()(image)

            #     # 保存图片
            #     save_path = os.path.join(save_folder, f'class_{class_idx}_sample_{i+1}.png')
            #     image.save(save_path)
            indices_all.extend(indices)

        print(indices_all)

        train_loader = data_utils.DataLoader(
            DatasetSplit(train_dataset, indices_all),
            batch_size=total_num,
            shuffle=False,
        )
        whole_img_dir = (
            f"/home/mwj/mycode2/Diffusion/Real_images/real_{ds}_{total_num}_train/"
        )
        os.makedirs(whole_img_dir, exist_ok=True)
        for i, (images, labels) in enumerate(train_loader):
            # 存成一张
            torchvision.utils.save_image(
                images,
                whole_img_dir + f"whole_{total_num}.png",
                nrow=num_of_per_class,
                normalize=True,
                scale_each=True,
            )
            # 存成单个图像
            for i, img in enumerate(images):
                class_dir = whole_img_dir + str(i // num_of_per_class)
                os.makedirs(class_dir, exist_ok=True)
                # print(img.shape)
                torchvision.utils.save_image(
                    img,
                    class_dir + f"/demo_{i%num_of_per_class}.png",
                    nrow=1,
                    normalize=True,
                    scale_each=True,
                )


# 由于一开始跑没修改完善代码，导致还是生成每个客户端10x10=100张图像，而MRI只有4个类，所以只需要生成40张
def get_random_40_images(root_dir):
    # 子目录列表（c_0 到 c_9）
    subdirs_clients = ["c_" + str(i) for i in range(num_clients)]
    sub_classes = [str(i) for i in range(num_classes)]
    # 每次从每个子目录中选择 10 张图片
    num_images_per_subdir = 10

    # 最终网格大小：4 行 10 列
    grid_rows = 4
    grid_cols = 10

    # 存储所有选中的图片张量
    image_tensors = []

    # 图像预处理：将 PIL 图像转换为张量
    transform = transforms.Compose([transforms.ToTensor()])  # 转换为张量，值范围 [0, 1]

    # 遍历每个子目录
    for subdir_c in subdirs_clients:
        subdir_path = os.path.join(root_dir, subdir_c)
        for sub_cls in sub_classes:
            subdir_path_detail = os.path.join(subdir_path, sub_cls)
            # 检查子目录是否存在
            if not os.path.exists(subdir_path_detail):
                print(f"子目录 {subdir_path_detail} 不存在，跳过...")
                continue

            # 获取子目录下的所有图片文件
            image_files = [
                f
                for f in os.listdir(subdir_path_detail)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]

            # 检查是否有足够的图片
            if len(image_files) < num_images_per_subdir:
                print(
                    f"子目录 {sub_cls} 中图片数量不足（{len(image_files)} < {num_images_per_subdir}），跳过..."
                )
                continue

            # 随机选择 10 张图片
            selected_images = random.sample(image_files, num_images_per_subdir)

            # 加载图片并转换为张量
            for img_name in selected_images:
                img_path = os.path.join(subdir_path_detail, img_name)
                try:
                    from PIL import Image

                    # 打开图片
                    img = Image.open(img_path).convert("RGB")  # 确保是 RGB 格式
                    # 转换为张量
                    img_tensor = transform(img)
                    image_tensors.append(img_tensor)
                except Exception as e:
                    print(f"加载图片 {img_path} 失败：{e}")
                    continue

        # 检查是否收集了足够的图片
        total_images = len(image_tensors)
        expected_images = grid_rows * grid_cols  # 4 * 10 = 40
        if total_images < expected_images:
            print(
                f"收集的图片数量不足（{total_images} < {expected_images}），无法生成 4x10 网格"
            )
            exit()

        # 如果图片数量超过 40，只取前 40 张
        image_tensors = image_tensors[:expected_images]

        # 检查所有图片的尺寸是否一致
        shapes = [img.shape for img in image_tensors]
        if len(set(shapes)) > 1:
            print("图片尺寸不一致，无法直接拼接！请确保所有图片尺寸相同。")
            print("图片尺寸：", shapes)
            exit()

        # 将图片张量列表转换为一个张量
        image_grid = torch.stack(image_tensors)  # 形状为 [40, C, H, W]

        # 保存图片
        output_path = os.path.join(subdir_path, "grid_4x10.png")
        vutils.save_image(
            image_grid,
            output_path,
            normalize=True,
            nrow=grid_cols,
            padding=2,
            pad_value=0,
            scale_each=True,
        )

        print(f"网格图片已保存到 {output_path}")


# pymao generate_real_imgs.py
if __name__ == "__main__":
    # 从MRI训练数据集中选择40个绘制4x25的网格
    # get_real_data()

    # 重新绘制，不要4x25的网格，随机选择40个生成4x10的网格
    # get_random_40_images(root_dir='/home/mwj/mycode2/Diffusion/results/2025_04_12_17_08_59_dif-iid-avgTrue-MRI-U10-F1-lrfc0.0002-E10-R300-BS256-tr0.8_s300/gen_images/")
    get_random_40_images(
        root_dir="results/2025_04_12_13_44_01_dif-iid2-da0.5-avgFalse-MRI-U10-F1-lrfc0.0002-E10-R300-BS256-tr0.8_s300/gen_images"
    )
