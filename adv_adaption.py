import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.resnet import resnet18
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score
from torchvision.models.feature_extraction import create_feature_extractor
import seaborn as sns
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

#28×28×1 --size of flattened image
flat_img = 784
datapath = '/data/mwj/data'
# 创建生成器和辨别器网络
input_channels = 1
latent_space = 64 # 512 loss变为nan
num_classes = 10
embedding_d = 128
batch_size = 128
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
method = 'TSNE' #'PCA'
# 辨别器的epoch
discr_e = 1
# 生成器没个round中的epoch
gen_e = 3
simclr_e = 1
temperature=0.5 # 0.3, 0.5, 0.7. (0.1和0.7的时候loss为nan)
adv_num_epoch = 150 # 150以内比较好，adv rounds(ar)
cls_epochs = 200 # classifier rounds(cr)
alpha = 0.5
dataset_name = 'mnist'
simclr_lr = 0.0001 # 0.001的时候loss为nan
gen_lr = 0.0002
disc_lr = 0.0002
cls_lr = 0.001

comments = f"{dataset_name}-ar{adv_num_epoch}-cr{cls_epochs}-ls{latent_space}-ed{embedding_d}-m{method}-de{discr_e}-ge{gen_e}-se{simclr_e}-t{temperature}-a{alpha}-bs{batch_size}-glr{gen_lr}-dlr{disc_lr}-clr{cls_lr}-slr{simclr_lr}"
print(comments)
result_name = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")+'_'+comments

# curr_working_dir = os.getcwd()
save_dir = os.path.join('./results', result_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

images_path = os.path.join(save_dir, 'gen_images')
log_name = os.path.join(save_dir, 'train.log')
if not os.path.exists(images_path):
    os.makedirs(images_path)

def set_logger(log_file_path="", file_name=""):

    # 创建一个 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个处理器，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 设置控制台日志级别

    # 创建一个处理器，用于将日志输出到文件

    file_handler = logging.FileHandler(log_file_path+file_name)
    file_handler.setLevel(logging.DEBUG)  # 设置文件日志级别

    # 创建一个格式化器，用于设置日志的格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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

curr_dir = './runs/' + result_name
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)

summary_writer = SummaryWriter(log_dir=curr_dir, comment=comments)

log.info(result_name)

# 计算距离矩阵
def compute_distances(features, prototypes):
    distances = torch.norm(torch.stack([f - prototypes for f in features]), dim=2)
    return torch.argmin(distances, dim=1)

def to_img(x):
    out = 0.5 * (x + 0.5)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


# 定义生成器网络
# 输入一个latent_space维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=16, dropout_rate=0.5):
        super(Generator, self).__init__()
        # 定义残差连接层
        self.residual = nn.Identity()
        self.fusion_img = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=2), # 16x14x14
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, 2 * output_channels, kernel_size=3, padding=1, stride=2), # 32x7x7
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2*output_channels),
            nn.Conv2d(2 * output_channels, 4 * output_channels, kernel_size=3, padding=1, stride=2), # 64x4x4
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.gen = nn.Sequential(
            nn.Linear(latent_space, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        self.linear = nn.Linear(1024, latent_space)

    def forward(self, x):
        # TODO 这时候直接输入随机噪声就行
        # out = self.gen(x)
        out = self.fusion_img(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.gen(out)
        return out

# 定义对比学习模型
class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(2 * 2 * 256, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_d)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        embeddings = self.projection_head(x)
        return x, embeddings

# 定义辨别器网络
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 512),  # 输入特征数为784，输出为512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(512, 256),  # 进行一个线性映射
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
            # 这里因为loss是用的nn.BCEWithLogitsLoss，所以不用写sigmoid
            # nn.Sigmoid()  # 也是一个激活函数，二分类问题中，sigmoid可以班实数映射到[0,1]，作为概率值，
        )

    def forward(self, x):
        x = self.dis(x)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, simclr_model, num_class=10):
        super(Classifier, self).__init__()
        # encoder
        self.encoder = simclr_model.encoder
        # classifier
        self.fc = nn.Linear(2 * 2 * 256, num_class, bias=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return feature, out

if dataset_name == 'mnist':
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root=datapath, train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root=datapath, train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

gen = Generator().to(device)

# 创建对比学习模型
simclr_model = SimCLR().to(device)
# print(simclr_model)

discriminator = Discriminator().to(device)


# 定义损失函数和优化器
criterion_gen = nn.BCEWithLogitsLoss()
criterion_disc = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(simclr_model.parameters(), lr=simclr_lr, weight_decay=5e-4)
# optimizer_gen = optim.Adam([
# {'params': gen.parameters()},
# {'params': simclr_model.parameters(), 'lr': 0.0001}], 0.00001)
optimizer_gen = optim.Adam(gen.parameters(), lr=gen_lr, weight_decay=5e-4)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=disc_lr, weight_decay=5e-4)

# 训练生成器和辨别器
gen.train()
simclr_model.train()
discriminator.train()

for epoch in range(adv_num_epoch):  # 迭代10次
    d_loss,g_loss,ls = 0,0,0
    for iter, (real_images, _) in enumerate(train_loader):
        batchsize = real_images.size(0)
        real_images = real_images.to(device)
        real_images.requires_grad_(False)
        

        # 真实样本的标签为1
        real_labels = torch.ones(batchsize, 1).to(device).detach()
        # 生成器生成样本的标签为0
        fake_labels = torch.zeros(batchsize, 1).to(device).detach()
        real_labels.requires_grad_(False)
        fake_labels.requires_grad_(False)

        for k in range(discr_e):
            # 训练辨别器
            optimizer_disc.zero_grad()

            real_outputs = discriminator(real_images.view(batchsize, -1))
            loss_real = criterion_disc(real_outputs, real_labels)
            real_scores = real_outputs  # 得到真实图片的判别值，输出的值越接近1越好
            # TODO Generator里面只用self.gen(x)的时候
            # fake_images = gen(torch.randn(batchsize, latent_space).to(device)).detach()
            fake_images = gen(real_images).detach()
            # print(fake_images[0])
            fake_outputs = discriminator(fake_images.to(device))  # detach生成的样本不参与生成器的梯度计算
            # print(fake_outputs)
            loss_fake = criterion_disc(fake_outputs, fake_labels)
            fake_scores = fake_outputs  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好

            loss_disc = (loss_real + loss_fake) / 2
            loss_disc.backward()
            optimizer_disc.step()
            # print(f'Epoch [{epoch+1}/10] [{k+1/str(discr_e)}], Loss Disc: {loss_disc.item()}')
        d_loss += loss_disc.item()

        # 训练生成器
        for i in range(gen_e):
            optimizer_gen.zero_grad()
            # TODO 
            # gen_images = gen(torch.randn(batchsize, latent_space).to(device))
            gen_images = gen(real_images)
            gen_outputs = discriminator(gen_images)
            loss_gen = criterion_gen(gen_outputs, real_labels)
            # loss_gen = loss_gen + loss
            # loss_gen = torch.log(1.0 - (discriminator(gen_images)).detach()) 
            loss_gen.backward()
            optimizer_gen.step()
        # print(f'Epoch [{epoch+1}/10] [{i+1/str(gen_e)}], Loss: {loss.item()}, Loss Gen: {loss_gen.item()}')
        g_loss += loss_gen.item()


        for k in range(simclr_e):
            optimizer.zero_grad()

            # 训练编码器
            fake_images = gen(real_images).detach()
            # print(real_images.size())
            # print(fake_images.size())
            # 计算原图和生成图像的表征
            _, embeddings_orig = simclr_model(real_images.view(batchsize, 1, 28, 28))
            _, embeddings_gen = simclr_model(fake_images.view(batchsize, 1, 28, 28))

            # 计算对比学习的损失
            # targets = torch.ones(embeddings_orig.size(0))
            # loss = criterion(embeddings_orig, embeddings_gen, targets)

            # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
            # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
            # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
            # 加上exp操作，该操作实际计算了分母
            # [2*B, D]
            out = torch.cat([embeddings_orig, embeddings_gen], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batchsize, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batchsize, -1)

            # 分子： *为对应位置相乘，也是点积
            # compute loss
            pos_sim = torch.exp(torch.sum(embeddings_orig * embeddings_gen, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            loss.backward()
            optimizer.step()
        ls += loss.item()

        # print(f'Epoch [{epoch+1}/10], Loss: {loss.item()}, Loss Gen: {loss_gen.item()}, Loss Disc: {loss_disc.item()}')
        # 打印中间的损失
        # print('Epoch[{}/{}], d_loss:{:.6f}, g_loss:{:.6f}, loss:{:.6f}'
        #       'D real: {:.6f}, D fake: {:.6f}'.format(epoch, num_epoch, loss_disc.data.item(), loss_gen.data.item(), loss.data.item(),
        #                                              real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
        #     ))
        if epoch == 0 and iter==len(train_loader)-1:
            real_images = to_img(real_images.cuda().data)
            save_image(real_images, os.path.join(images_path, 'r'+str(adv_num_epoch)+'_real_images.png'))
        if iter==len(train_loader)-1:
            fake_images = to_img(fake_images.cuda().data)
            save_image(fake_images, os.path.join(images_path, 'r'+str(adv_num_epoch)+'_fake_images-{}.png'.format(epoch + 1)))
    summary_writer.add_scalar('Train-Adv/d_loss', d_loss/len(train_loader), epoch+1)
    summary_writer.add_scalar('Train-Adv/g_loss', g_loss/len(train_loader), epoch+1)
    summary_writer.add_scalar('Train-Adv/loss', ls/len(train_loader), epoch+1)
    # 这里前几次跑，要么是两个都是real_scores，要么两个写反了
    summary_writer.add_scalar('Train-Adv/real_scores', real_scores.data.mean().item(), epoch+1)
    summary_writer.add_scalar('Train-Adv/fake_scores', fake_scores.data.mean().item(), epoch+1)
    log.info('Epoch[{}/{}], d_loss:{:.6f}, g_loss:{:.6f}, loss:{:.6f}, D real: {:.6f}, D fake: {:.6f}'.format(epoch, adv_num_epoch,
        d_loss/len(train_loader), g_loss/len(train_loader), ls/len(train_loader), real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
        ))

# 测试生成器网络
gen.eval()

simclr_save_path = os.path.join(save_dir, str(adv_num_epoch)+'_simclr.pth')
gen_save_path = os.path.join(save_dir, str(adv_num_epoch)+'_gen.pth')

torch.save(simclr_model.state_dict(), simclr_save_path)
torch.save(gen.state_dict(), gen_save_path)


# simclr_model = SimCLR().to(device)
# simclr_model.load_state_dict(torch.load(simclr_save_path))

# gen = Generator().to(device)
# gen.load_state_dict(torch.load(gen_save_path))

# 训练完encoder之后
# 初始化原型矩阵
prototypes = torch.zeros(10, 1024)
# 统计每个类别的样本数量
class_counts = torch.zeros(10)

# 遍历整个数据集
for images, labels in train_loader:
    images = images.to(device)
    feature, _ = simclr_model(images)
    feature = feature.cpu().detach().numpy()
    for i in range(10):  # 遍历每个类别
        class_indices = (labels == i)  # 找到属于当前类别的样本的索引
        class_outputs = feature[class_indices]  # 提取属于当前类别的样本的特征向量
        prototypes[i] += np.sum(class_outputs, axis=0)  # 将当前类别的特征向量累加到原型矩阵中
        class_counts[i] += class_outputs.shape[0]  # 统计当前类别的样本数量

# 计算每个类别的平均特征向量
for i in range(10):
    if class_counts[i] > 0:
        prototypes[i] /= class_counts[i]

log.info(f"Prototypes computed successfully! {prototypes}")


clser = Classifier(simclr_model).to(device)
optimizer_cls = optim.Adam(clser.fc.parameters(), lr=cls_lr, weight_decay=5e-4)
# 定义交叉熵损失函数和优化器
CE_criterion = nn.CrossEntropyLoss()

prototypes = prototypes.to(device)
prototypes.requires_grad_(False)
# 训练分类器
clser.train()
for epo in range(cls_epochs):
    running_loss_t, running_loss_f = 0.0, 0.0
    loss_ce = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_cls.zero_grad()
        # 假样本
        fake_imgs = gen(images).detach()
        features_f, outputs_f = clser(fake_imgs.view(images.size(0), 1, 28, 28))
        pseudo_labels_f = compute_distances(features_f, prototypes)
        # similarity_scores_f = torch.matmul(features_f, prototypes.t())  # 计算相似度(效果很差)
        # _, pseudo_labels_f = torch.max(similarity_scores_f, dim=1)  # 选择最相似的类别作为伪标签
        pseudo_labels_f.requires_grad_(False)
        pseudo_labels_f = pseudo_labels_f.detach()
        loss_ce_f = CE_criterion(outputs_f, pseudo_labels_f)  # 使用伪标签计算交叉熵损失
        running_loss_f += loss_ce_f.item() * images.size(0)
        # 真样本
        features_t, outputs = clser(images)
        # print(outputs)
        loss_ce_true = CE_criterion(outputs, labels)  # 使用logits计算交叉熵损失
        loss_ce = alpha * loss_ce_f + (1-alpha) * loss_ce_true
        loss_ce.backward()
        optimizer_cls.step()
        running_loss_t += loss_ce_true.item() * images.size(0)
    summary_writer.add_scalar('Train-cls/t_loss', running_loss_t / len(train_dataset), epo+1)
    summary_writer.add_scalar('Train-cls/f_loss', running_loss_f / len(train_dataset), epo+1)

    log.info('Epoch [%d/%d], Loss_t: %.4f, Loss_f: %.4f' % (epo+1, cls_epochs, running_loss_t / len(train_dataset), running_loss_f / len(train_dataset)))


# 测试编码器的准确率
clser.eval()
with torch.no_grad():
    true_labels = []
    pred_labels = []
    out_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        features, outputs = clser(images)
        pred = compute_distances(features, prototypes)
        # similarity_scores = torch.matmul(features, prototypes.t())  # 计算相似度(效果不如L2)
        # _, pred = torch.max(similarity_scores, dim=1)  # 选择最相似的类别作为预测标签
        _, outd = torch.max(outputs, dim=1)
        true_labels.extend(labels.numpy())
        pred_labels.extend(pred.cpu().numpy())
        out_labels.extend(outd.cpu().numpy())
accuracy1 = accuracy_score(true_labels, pred_labels)
log.info(f'pseudo Accuracy: {accuracy1}') # pseudo Accuracy: 0.4556
accuracy2 = accuracy_score(true_labels, out_labels)
log.info(f'test Accuracy: {accuracy2}') # Accuracy: 0.7762

cls_save_path = os.path.join(save_dir, str(adv_num_epoch)+'_clser.pth')

torch.save(clser.state_dict(), cls_save_path)

# clser.load_state_dict(torch.load(str(adv_num_epoch)+'_clser.pth'))

model_trunc = create_feature_extractor(clser, return_nodes={'encoder': 'semantic_feature'})

data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
encoding_array = []
labels_list = []
for batch_idx, (images, labels) in enumerate(data_loader):
    images, labels = images.to(device), labels.to(device)
    labels_list.append(labels.item())
    feature = model_trunc(images)['semantic_feature'].squeeze().flatten().detach().cpu().numpy() # 执行前向预测，得到 avgpool 层输出的语义特征
    encoding_array.append(feature)
encoding_array = np.array(encoding_array)
# 保存为本地的 npy 文件
np.save(os.path.join(save_dir, 'clser测试集语义特征_mnist.npy'), encoding_array)


marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_list = test_dataset.classes
n_class = len(class_list) # 测试集标签类别数
palette = sns.hls_palette(n_class) # 配色方案
sns.palplot(palette)
# 随机打乱颜色列表和点型列表
random.seed(1234)
random.shuffle(marker_list)
random.shuffle(palette)


for method in ['PCA', 'TSNE']:
    #选择降维方法
    if method == 'PCA': 
        X_2d = PCA(n_components=2).fit_transform(encoding_array)
    if method == 'TSNE': 
        X_2d = TSNE(n_components=2, random_state=0, n_iter=20000).fit_transform(encoding_array)

    class_to_idx = test_dataset.class_to_idx

    plt.figure(figsize=(14, 14))
    for idx, fruit in enumerate(class_list): # 遍历每个类别
        #print(fruit)
        # 获取颜色和点型
        color = palette[idx]
        marker = marker_list[idx%len(marker_list)]
        # 找到所有标注类别为当前类别的图像索引号
        indices = np.where(np.array(labels_list)==class_to_idx[fruit])
        plt.scatter(X_2d[indices, 0], X_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)
    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])

    dim_reduc_save_path = os.path.join(save_dir, f'clser_mnist_语义特征{method}二维降维可视化.pdf')

    plt.savefig(dim_reduc_save_path, dpi=300, bbox_inches='tight') # 保存图像

log.info(result_name)