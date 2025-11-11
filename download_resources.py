"""
下载并保存数据集和模型到本地
运行此脚本需要联网,只需运行一次
"""

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from tqdm import tqdm
import time

# 设置本地缓存目录
CACHE_DIR = "./cache"
DATASET_DIR = os.path.join(CACHE_DIR, "datasets")
MODEL_DIR = os.path.join(CACHE_DIR, "models")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def print_header(text):
    """打印美化的标题"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(step_num, total_steps, text):
    """打印步骤信息"""
    print(f"\n{'█' * 70}")
    print(f"  步骤 [{step_num}/{total_steps}]: {text}")
    print(f"{'█' * 70}")

def print_info(text, emoji="ℹ️"):
    """打印提示信息"""
    print(f"\n{emoji}  {text}")

def print_success(text):
    """打印成功信息"""
    print(f"✅ {text}")

def print_warning(text):
    """打印警告信息"""
    print(f"⚠️  {text}")

print_header("📥 资源下载工具")
print_info("此工具将下载训练所需的数据集和模型", "🎯")
print_warning("请确保网络连接正常,下载过程可能需要几分钟")
print_info(f"下载目录: {os.path.abspath(CACHE_DIR)}", "📁")

start_time = time.time()

# 1. 下载数据集
print_step(1, 3, "下载数据集")
print_info("数据集: frgfm/imagenette (ImageNet 的简化版本)", "📊")
print_info("包含 10 个类别的图像数据", "🏷️")

try:
    # 使用 320px 配置平衡速度和质量
    dataset = load_dataset("frgfm/imagenette", "full_size")
    dataset_path = os.path.join(DATASET_DIR, "imagenette")
    dataset.save_to_disk(dataset_path)
    
    # 显示数据集信息
    total_samples = sum(len(dataset[split]) for split in dataset.keys())
    print_info(f"数据集统计:", "📈")
    for split in dataset.keys():
        print(f"  • {split}: {len(dataset[split])} 样本")
    print(f"  • 总计: {total_samples} 样本")
    
    print_success(f"数据集下载完成,保存至: {dataset_path}")
except Exception as e:
    print_warning(f"数据集下载失败: {e}")
    exit(1)

# 2. 下载模型和处理器
model_name = "microsoft/resnet-50"
print_step(2, 3, "下载图像处理器")
print_info(f"模型: {model_name}", "🤖")

try:
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    processor_path = os.path.join(MODEL_DIR, "resnet-50")
    image_processor.save_pretrained(processor_path)
    print_success(f"图像处理器下载完成,保存至: {processor_path}")
except Exception as e:
    print_warning(f"图像处理器下载失败: {e}")
    exit(1)

print_step(3, 3, "下载预训练模型")
print_info("ResNet-50: 50层残差网络,约 25M 参数", "🧠")
print_warning("模型文件较大 (~100MB),请耐心等待...")

try:
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model_path = os.path.join(MODEL_DIR, "resnet-50")
    model.save_pretrained(model_path)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print_info(f"模型参数量: {total_params:,}", "📊")
    print_success(f"预训练模型下载完成,保存至: {model_path}")
except Exception as e:
    print_warning(f"预训练模型下载失败: {e}")
    exit(1)

end_time = time.time()
duration = end_time - start_time

print_header("✨ 所有资源下载完成!")
print_info(f"总用时: {duration:.1f} 秒", "⏱️")
print_info(f"数据集位置: {os.path.abspath(DATASET_DIR)}", "📁")
print_info(f"模型位置: {os.path.abspath(MODEL_DIR)}", "📁")
print_info("\n下一步:", "🎯")
print("  运行以下命令开始训练:")
print("  python main.py  (或 python main_offline.py)")
print("\n  提示: 现在可以完全离线运行训练!")
