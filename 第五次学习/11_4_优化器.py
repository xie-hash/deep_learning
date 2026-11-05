# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torch import nn


# #导入数据
# data_transforms = datasets.CIFAR10(r"C:\Users\谢\Desktop\vscodepj\深度学习\第三次学习\dataset", train=True, download=True, transform=transforms.ToTensor())
# data_loader = DataLoader(data_transforms,batch_size=32,shuffle=True,num_workers=3)

# class xie_nn(nn.Module):
#     def __init__(self):
#         super(xie_nn,self).__init__()
#         self.module = nn.Sequential(
#                         nn.Conv2d(3,32,5,stride=1,padding=2,dilation=1,bias=True),
#             nn.MaxPool2d(kernel_size=2,padding=0,stride=2),
#             nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2,dilation=1,bias=True),
#             nn.MaxPool2d(kernel_size=2,padding=0,stride=2), 
#             nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2,dilation=1,bias=True),
#             nn.MaxPool2d(kernel_size=2,padding=0,stride=2),
#             nn.Flatten(),
#             nn.Linear(in_features=1024,out_features=64,bias=True),
#             nn.Linear(in_features=64,out_features=10,bias=True)
#         )
#     def forward(self,x):
#         x = self.module(x)
#         return x
# model = xie_nn()
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=xie_nn().parameters(),lr=0.03,momentum=0.9)


# for epoch in range(20):
#     loss_sum = 0
#     for batch in data_loader:
#         data, target = batch
#         output = model(data)
#         loss_value = loss(output, target)
#         optimizer.zero_grad()
#         loss_value.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         loss_sum += loss_value.item()
#     print("epoch:", epoch, "loss:", loss_sum)


    
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ---------- 数据 ----------
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(
    root=r"C:\Users\谢\Desktop\vscodepj\深度学习\第三次学习\dataset",
    train=True, download=True, transform=data_transforms
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------- 模型 ----------
class xie_nn(nn.Module):
    def __init__(self):
        super(xie_nn, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, 5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(1024, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )

    def forward(self, x):
        return self.module(x)
    
class xie_nn_2(nn.Module):
    def __init__(self):
        super(xie_nn_2, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True),

            nn.Flatten(),
            nn.Linear(1024, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )

    def forward(self, x):
        return self.module(x)
    

# ---------- 模型、损失、优化器 ----------
model = xie_nn_2().to(device)
criterion = nn.CrossEntropyLoss()

# 只有优化器，没有 scheduler 了。学习率固定为 0.03
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
print(model)
model.add_module("linearn", nn.Linear(64, 10))
print(model)
torch.save(model,'model.pth')

# ---------- 训练 ----------
# EPOCHS = 20
# for epoch in range(EPOCHS):
#     model.train()  # 告诉模型现在是在训练模式（虽然目前没 Dropout 和 BN，但养成好习惯）
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     start = time.time()

#     # tqdm 进度条
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)
#     for data, target in pbar:
#         data, target = data.to(device), target.to(device)

#         # 1. 前向传播
#         output = model(data)
#         loss_value = criterion(output, target)

#         # 2. 清空梯度、反向传播、更新参数
#         optimizer.zero_grad()
#         loss_value.backward()
#         optimizer.step()

#         # 3. 统计准确率和损失
#         bs = data.size(0)
#         total_loss += loss_value.item() * bs
#         pred = output.argmax(dim=1)
#         total_correct += (pred == target).sum().item()
#         total_samples += bs

#         # 更新进度条上的信息
#         pbar.set_postfix({
#             "loss": f"{loss_value.item():.4f}",
#             "acc": f"{total_correct / total_samples * 100:.1f}%",
#         })

#     # 这个 epoch 结束了，打印总结信息
#     avg_loss = total_loss / total_samples
#     acc = total_correct / total_samples * 100
    
#     # 注意这里：学习率现在固定是 0.03，不会变了
#     print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, "
#           f"train_acc={acc:.2f}%, "
#           f"lr={optimizer.param_groups[0]['lr']:.4f}, " # 这句打印代码留着，方便你以后加 scheduler 时观察
#           f"time={time.time()-start:.1f}s")