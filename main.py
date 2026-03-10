import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# ===========================
# 配置类
# ===========================
class Config:
    """训练过程的配置参数。"""
    data_dir = os.path.join(os.path.dirname(__file__), "datasets")  # 数据集目录
    batch_size = 32                      # 每批次的样本数量
    num_workers = 4                      # 数据加载的子进程数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    epochs = 50                          # 训练的总轮数
    learning_rate = 0.001                # 优化器的学习率
    weight_decay = 1e-4                  # 优化器的权重衰减（L2正则化）
    num_classes = 2                      # 输出类别数量
    input_size = 224                     # 输入图像的大小（高度和宽度）

# ===========================
# 自数据集类
# ===========================
class BreastCancerDataset(Dataset):
    """
    乳腺癌图像的数据集类。
    
    参数:
        images (list): 图像文件路径列表。
        labels (list): 对应的标签列表。
        transform (callable, optional): 可选的图像变换。
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """返回数据集的样本总数。"""
        return len(self.images)

    def __getitem__(self, idx):
        """获取指定索引的图像和标签。"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 打开图像并转换为RGB
        image = Image.open(img_path).convert('RGB')
        
        # 如果有变换，则应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ===========================
# 注意力模块
# ===========================
class AttentionModule(nn.Module):
    """
    简单的注意力模块，使用卷积层实现。
    
    参数:
        in_channels (int): 输入通道数。
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """将注意力权重应用于输入特征图。"""
        attention_weights = self.attention(x)
        return x * attention_weights

# ===========================
# CNN模型
# ===========================
class CNN(nn.Module):
    """
    卷积神经网络模型。
    
    参数:
        num_classes (int): 输出类别数。
        use_attention (bool): 是否使用注意力模块。
    """
    def __init__(self, num_classes=2, use_attention=False):
        super(CNN, self).__init__()
        self.use_attention = use_attention
        
        # 定义CNN的特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 卷积层1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 最大池化
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 卷积层2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 最大池化
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 卷积层3
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # 最大池化
        )
        
        # 可选的注意力模块
        if use_attention:
            self.attention = AttentionModule(256)
            
        # 定义CNN的分类器部分
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),                # 自适应平均池化
            nn.Flatten(),                                # 展平张量
            nn.Linear(256, num_classes)                  # 全连接层
        )

    def forward(self, x):
        """定义CNN的前向传播。"""
        x = self.features(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.classifier(x)
        return x

# ===========================
# ResNet基础块
# ===========================
class ResBlock(nn.Module):
    """
    ResNet的基础残差块。
    
    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        stride (int): 第一个卷积层的步幅。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 如果维度变化，通过卷积匹配维度
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """定义残差块的前向传播。"""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 添加快捷连接
        out = torch.relu(out)
        return out

# ===========================
# ResNet模型
# ===========================
class ResNet(nn.Module):
    """
    ResNet模型。
    
    参数:
        num_classes (int): 输出类别数。
        use_attention (bool): 是否使用注意力模块。
    """
    def __init__(self, num_classes=2, use_attention=False):
        super(ResNet, self).__init__()
        self.use_attention = use_attention
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self.make_layer(64, 64, 2)      # 第一层
        self.layer2 = self.make_layer(64, 128, 2, stride=2)  # 第二层
        self.layer3 = self.make_layer(128, 256, 2, stride=2) # 第三层
        
        # 可选的注意力模块
        if use_attention:
            self.attention = AttentionModule(256)
            
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        多个残差块的层。
        
        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            blocks (int): 残差块的数量。
            stride (int): 第一个残差块的步幅。
        
        返回:
            nn.Sequential: 由多个残差块组成的顺序容器。
        """
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """定义ResNet的前向传播。"""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===========================
# 训练函数
# ===========================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    训练并验证模型。
    
    参数:
        model (nn.Module): 要训练的神经网络模型。
        train_loader (DataLoader): 训练数据的DataLoader。
        val_loader (DataLoader): 验证数据的DataLoader。
        criterion (nn.Module): 损失函数。
        optimizer (optim.Optimizer): 优化器。
        num_epochs (int): 训练的轮数。
        device (torch.device): 训练设备。
    
    返回:
        tuple: 训练损失列表，训练准确率列表，验证损失列表，验证准确率列表，最佳验证指标字典。
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 跟踪最佳验证准确率
    best_val_acc = 0.0
    best_metrics = {}

    for epoch in range(num_epochs):
        # ===========================
        # 训练阶段
        # ===========================
        model.train()  # 设置模型为训练模式
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 训练进度条
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()            # 清除梯度
            outputs = model(images)          # 前向传播
            loss = criterion(outputs, labels)# 计算损失
            
            loss.backward()                  # 反向传播
            optimizer.step()                 # 更新参数
            
            train_loss += loss.item()        # 累积损失
            _, predicted = outputs.max(1)    # 获取预测类别
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条显示的损失和准确率
            train_bar.set_postfix({
                'Loss': f'{train_loss/len(train_loader):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 计算本轮的平均训练损失和准确率
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100. * correct / total)
        
        # ===========================
        # 验证阶段
        # ===========================
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # 禁用梯度计算
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)          # 前向传播
                loss = criterion(outputs, labels)# 计算损失
                
                val_loss += loss.item()          # 累积损失
                _, predicted = outputs.max(1)    # 获取预测类别
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条显示的损失和准确率
                val_bar.set_postfix({
                    'Loss': f'{val_loss/len(val_loader):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        # 计算本轮的平均验证损失和准确率
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100. * correct / total)
        
        # ===========================
        # 记录最佳验证指标
        # ===========================
        # 如果当前验证准确率是最好的，则记录下来
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_metrics = {
                'Epoch': epoch + 1,
                'Train Loss': train_losses[-1],
                'Train Acc': train_accs[-1],
                'Val Loss': val_losses[-1],
                'Val Acc': val_accs[-1]
            }
        
        # ===========================
        # 打印本轮摘要
        # ===========================
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%')
        print(f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%')
        print('-' * 50)
    
    return train_losses, train_accs, val_losses, val_accs, best_metrics

# ===========================
# 绘图函数
# ===========================
def plot_training(train_losses, train_accs, val_losses, val_accs, model_name):
    """
    绘制训练和验证的损失与准确率曲线。
    
    参数:
        train_losses (list): 每轮的训练损失列表。
        train_accs (list): 每轮的训练准确率列表。
        val_losses (list): 每轮的验证损失列表。
        val_accs (list): 每轮的验证准确率列表。
        model_name (str): 模型名称（用于图表标题和保存文件名）。
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'{model_name} - 损失')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.title(f'{model_name} - 准确率')
    plt.xlabel('轮数')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png')  # 保存为图片文件
    plt.close()

# ===========================
# 主函数
# ===========================
def main():
    """主函数，执行训练流程。"""
    config = Config()
    
    # ===========================
    # 数据预处理
    # ===========================
    transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),  # 调整图像大小
        transforms.ToTensor(),                                     # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],          # 使用ImageNet的均值进行归一化
                             std=[0.229, 0.224, 0.225])           # 使用ImageNet的标准差进行归一化
    ])
    
    # ===========================
    # 数据加载
    # ===========================
    images = []
    labels = []
    for class_idx, class_name in enumerate(['benign', 'malignant']):
        class_dir = os.path.join(config.data_dir, class_name)  # 类别目录路径
        for img_name in os.listdir(class_dir):
            # 检查有效的图像文件扩展名
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                images.append(os.path.join(class_dir, img_name))  # 添加图像路径
                labels.append(class_idx)                          # 添加对应标签
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据集
    train_dataset = BreastCancerDataset(X_train, y_train, transform)
    val_dataset = BreastCancerDataset(X_val, y_val, transform)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers
    )
    
    # ===========================
    # 模型初始化
    # ===========================
    models = {
        'CNN': CNN(config.num_classes, use_attention=False),
        'CNN_Attention': CNN(config.num_classes, use_attention=True),
        'ResNet': ResNet(config.num_classes, use_attention=False),
        'ResNet_Attention': ResNet(config.num_classes, use_attention=True)
    }
    
    # 存储每个模型的最佳指标
    all_best_metrics = []
    
    # ===========================
    # 对每个模型进行训练
    # ===========================
    for model_name, model in models.items():
        print(f'\n正在训练模型: {model_name}...')
        model = model.to(config.device)                  
        criterion = nn.CrossEntropyLoss()                # 定义损失函数
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )                                                 # 定义优化器
        
        # 训练模型
        train_losses, train_accs, val_losses, val_accs, best_metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            config.epochs, config.device
        )
        
        # 保存模型的状态字典
        torch.save(model.state_dict(), f'{model_name}.pth')
        
        plot_training(train_losses, train_accs, val_losses, val_accs, model_name)
        
        best_metrics['Model'] = model_name
        all_best_metrics.append(best_metrics)
    
    # ===========================
    # 将最佳指标保存到CSV
    # ===========================
    csv_file = 'best_metrics.csv'
    csv_columns = ['Model', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc']
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader() 
            for data in all_best_metrics:
                writer.writerow(data)  
        print(f'\n所有模型的最佳指标已保存到 {csv_file}')
    except IOError:
        print("写入CSV文件时发生I/O错误")


if __name__ == '__main__':
    main()
