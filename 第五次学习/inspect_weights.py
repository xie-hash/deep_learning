"""
查看模型参数的小工具
用法:
    python inspect_weights.py                 # 默认读 ../../model.pth
    python inspect_weights.py xxx.pth         # 指定文件
"""
import sys
import torch
import torch.nn as nn


# 必须和保存时完全一致的结构，否则 load_state_dict 会报 size mismatch
class xie_nn_2(nn.Module):
    def __init__(self):
        super(xie_nn_2, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(1024, 64, bias=True), nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )

    def forward(self, x):
        return self.module(x)


def load_any(path):
    """兼容两种保存方式: 整个模型 / state_dict"""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj, "整个模型对象"
    model = xie_nn_2()
    model.load_state_dict(obj)
    return model, "state_dict（已灌入新建模型）"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\谢\Desktop\vscodepj\model.pth"
    model, how = load_any(path)

    print("=" * 78)
    print("文件 :", path)
    print("格式 :", how)
    print("参数 :", f"{sum(p.numel() for p in model.parameters()):,}")
    print("=" * 78)
    print(f"{'layer':<20}{'shape':<20}{'params':>8}{'mean':>10}{'std':>10}{'min':>9}{'max':>9}{'zero%':>7}")
    print("-" * 78)

    for name, p in model.named_parameters():
        d = p.detach()
        print(f"{name:<20}{str(tuple(d.shape)):<20}{d.numel():>8,}"
              f"{d.mean():>10.4f}{d.std():>10.4f}{d.min():>9.3f}{d.max():>9.3f}"
              f"{(d == 0).float().mean() * 100:>6.1f}%")

    print("-" * 78)
    conv1 = model.module[0].weight.detach()
    print("第一个卷积层 module.0.weight，形状", tuple(conv1.shape), "= [输出通道, 输入通道, 高, 宽]")
    print("取第 0 个输出通道、第 0 个输入通道的 5x5 卷积核：")
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=100)
    print(conv1[0, 0])


if __name__ == "__main__":
    main()
