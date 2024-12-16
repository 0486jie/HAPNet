import random
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 加载数据
hsi_data = scipy.io.loadmat('D:\jieprojects\MultiModal\data\Augsburg\\augsburg_hsi.mat')['augsburg_hsi']
sar_data = scipy.io.loadmat('D:\jieprojects\MultiModal\data\Augsburg\\augsburg_sar.mat')['augsburg_sar']
gt_data = scipy.io.loadmat('D:\jieprojects\MultiModal\data\Augsburg\\augsburg_gt.mat')['augsburg_gt'] - 1
index_data = scipy.io.loadmat('D:\jieprojects\MultiModal\data\Augsburg\\augsburg_index.mat')

train_indices = index_data['augsburg_train']
test_indices = index_data['augsburg_test']


# 数据预处理
def preprocess_data(data, n_components):
    num_pixels = data.shape[0] * data.shape[1]
    num_bands = data.shape[2]
    reshaped_data = data.reshape((num_pixels, num_bands))
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(reshaped_data)
    pca_data = pca_data.reshape(data.shape[0], data.shape[1], n_components)
    return pca_data / np.max(pca_data)


hsi_data_pca = preprocess_data(hsi_data, n_components=30)
sar_data_normalized = sar_data / np.max(sar_data)

# 标签编码
label_encoder = LabelEncoder()
gt_data_flat = gt_data.flatten()
gt_data_encoded = label_encoder.fit_transform(gt_data_flat).reshape(gt_data.shape)


# 数据集定义
class MultiModalDataset(Dataset):
    def __init__(self, hsi_data, sar_data, gt_data, indices, window_size=7, augment=False):
        self.hsi_data = hsi_data
        self.sar_data = sar_data
        self.gt_data = gt_data
        self.indices = [
            (row, col) for row, col in indices
            if (window_size // 2 <= row < hsi_data.shape[0] - window_size // 2 and
                window_size // 2 <= col < hsi_data.shape[1] - window_size // 2)
        ]
        self.window_size = window_size
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row, col = self.indices[idx]
        half_window = self.window_size // 2
        hsi_window = self.hsi_data[row - half_window:row + half_window + 1,
                     col - half_window:col + half_window + 1, :]
        sar_window = self.sar_data[row - half_window:row + half_window + 1,
                     col - half_window:col + half_window + 1, :]
        label = self.gt_data[row, col]
        hsi_window = torch.tensor(hsi_window, dtype=torch.float32).permute(2, 0, 1)
        sar_window = torch.tensor(sar_window, dtype=torch.float32).permute(2, 0, 1)
        return hsi_window, sar_window, torch.tensor(label, dtype=torch.long)


train_dataset = MultiModalDataset(hsi_data_pca, sar_data_normalized, gt_data_encoded, train_indices, augment=True)
test_dataset = MultiModalDataset(hsi_data_pca, sar_data_normalized, gt_data_encoded, test_indices, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义锚定自注意力模块
class AnchoredSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(AnchoredSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.anchor_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        anchor = self.anchor_conv(x)

        # 计算注意力权重
        attention_weights = torch.matmul(query.view(query.size(0), query.size(1), -1).transpose(1, 2),
                                         key.view(key.size(0), key.size(1), -1))
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 锚定自注意力
        attended = torch.matmul(attention_weights, value.view(value.size(0), value.size(1), -1).transpose(1, 2))
        attended = attended.transpose(1, 2).view(x.size())

        return x + attended + anchor


# 定义Hierarchical Attention Module (HAM)
class HierarchicalAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(HierarchicalAttentionModule, self).__init__()
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spectral_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0)),
            nn.Sigmoid()
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.asa = AnchoredSelfAttention(in_channels)

    def forward(self, x):
        global_att = self.global_branch(x)
        spectral_att = self.spectral_branch(x)
        local_feat = self.local_branch(x)
        attended_feat = self.asa(x)
        return x * global_att + x * spectral_att + local_feat + attended_feat


# 定义Parallel Filter Fusion Module (PFFM)
class ParallelFilterFusionModule(nn.Module):
    def __init__(self, hsi_channels, sar_channels):
        super(ParallelFilterFusionModule, self).__init__()
        self.hsi_fft_filter = nn.Sequential(
            nn.Conv2d(hsi_channels, hsi_channels, kernel_size=1),
            nn.ReLU()
        )
        self.sar_fft_filter = nn.Sequential(
            nn.Conv2d(sar_channels, sar_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, hsi, sar):
        hsi_fft = torch.fft.fft2(hsi)
        sar_fft = torch.fft.fft2(sar)

        hsi_filtered = torch.fft.ifft2(hsi_fft * self.hsi_fft_filter(hsi).real)
        sar_filtered = torch.fft.ifft2(sar_fft * self.sar_fft_filter(sar).real)

        return hsi_filtered.real + sar_filtered.real


# 定义主模型
class HAPNet(nn.Module):
    def __init__(self, hsi_channels=30, sar_channels=4, num_classes=7):
        super(HAPNet, self).__init__()
        self.hsi_conv = nn.Conv2d(hsi_channels, 64, kernel_size=3, padding=1)
        self.sar_conv = nn.Conv2d(sar_channels, 64, kernel_size=3, padding=1)
        self.ham = HierarchicalAttentionModule(64)
        self.pffm = ParallelFilterFusionModule(64, 64)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, hsi, sar):
        hsi_feat = self.hsi_conv(hsi)
        sar_feat = self.sar_conv(sar)
        feat = self.pffm(hsi_feat, sar_feat)
        feat = self.ham(feat)
        return self.fc(feat)


# 模型训练和评估函数保持一致
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for hsi, sar, labels in tqdm(train_loader):
        hsi, sar, labels = hsi.to(device), sar.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(hsi, sar)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for hsi, sar, labels in tqdm(test_loader):
            hsi, sar, labels = hsi.to(device), sar.to(device), labels.to(device)
            outputs = model(hsi, sar)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# 初始化模型、损失函数和优化器
model = HAPNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练和评估循环
num_epochs = 15
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    test_accuracy = test(model, test_loader, device)

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
