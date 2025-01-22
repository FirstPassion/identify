import os
import random
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool, cpu_count  # 导入多进程模块

# 设置随机种子以确保可重复性
random.seed(42)

# 数据增强的变换
data_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(30),  # 随机旋转30度
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # 颜色抖动
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
    ]
)

# 原始数据路径
data_dir = "data/train"
# 输出路径
output_dir = "data_augmented"
os.makedirs(output_dir, exist_ok=True)


# 定义一个函数来处理单个类别的图片
def process_class(class_name):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_dir):
        return

    # 创建输出目录
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [
        f for f in os.listdir(class_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    # 对每张图片进行数据增强
    for image_file in image_files:
        image_path = os.path.join(class_dir, image_file)
        image = Image.open(image_path)

        # 如果图片是 RGBA 模式，转换为 RGB 模式
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # 保存原始图片
        original_image_path = os.path.join(output_class_dir, f"original_{image_file}")
        image.save(original_image_path)

        # 生成增强后的图片
        for i in range(5):  # 每张图片生成5张增强后的图片
            augmented_image = data_transforms(image)
            augmented_image_path = os.path.join(
                output_class_dir, f"augmented_{i}_{image_file}"
            )
            augmented_image.save(augmented_image_path)


# 使用多进程处理所有类别
def process_all_classes():
    class_names = [
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    with Pool(processes=cpu_count()) as pool:  # 使用所有可用的 CPU 核心
        list(
            tqdm(
                pool.imap(process_class, class_names),
                total=len(class_names),
                desc="处理类别",
            )
        )


# 划分训练、验证和测试集
def split_dataset():
    all_images = []
    all_labels = []

    for class_name in tqdm(os.listdir(output_dir), desc="收集图片和标签"):
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 获取所有图片文件
        image_files = [
            f for f in os.listdir(class_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        all_images.extend([os.path.join(class_dir, f) for f in image_files])
        all_labels.extend([class_name] * len(image_files))

    # 划分数据集：60% 训练集，20% 验证集，20% 测试集
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.25, random_state=42
    )  # 0.25 x 0.8 = 0.2

    # 创建训练、验证和测试集目录
    for dataset in ["train", "val", "test"]:
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        for class_name in set(all_labels):
            os.makedirs(os.path.join(dataset_dir, class_name), exist_ok=True)

    # 将图片移动到相应的目录
    for image, label in tqdm(zip(train_images, train_labels), desc="移动训练集图片"):
        os.rename(
            image, os.path.join(output_dir, "train", label, os.path.basename(image))
        )

    for image, label in tqdm(zip(val_images, val_labels), desc="移动验证集图片"):
        os.rename(
            image, os.path.join(output_dir, "val", label, os.path.basename(image))
        )

    for image, label in tqdm(zip(test_images, test_labels), desc="移动测试集图片"):
        os.rename(
            image, os.path.join(output_dir, "test", label, os.path.basename(image))
        )


# 主函数
def main():
    # 使用多进程处理所有类别的图片
    process_all_classes()

    # 划分数据集
    split_dataset()

    print("数据增强和划分完成！")


if __name__ == "__main__":
    main()
