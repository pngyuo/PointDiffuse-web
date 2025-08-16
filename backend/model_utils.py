import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import time


# 伪模型类，用于在没有真实模型时提供基本功能
class DummyModel:
    def __init__(self, device):
        self.device = device
        self.num_classes = 13  # S3DIS默认13类

    def predict(self, points_tensor, features_tensor):
        # 生成伪预测结果
        points = points_tensor.cpu().numpy()[0]
        num_points = points.shape[0]

        # 随机生成预测结果
        predictions = np.random.randint(0, self.num_classes, num_points)

        # 生成伪logits
        logits = np.random.rand(num_points, self.num_classes)

        return predictions, logits


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.getenv('MODEL_PATH', '/app/model/best_model.pth')

    if os.path.exists(model_path):
        # 真实模型加载逻辑
        try:
            # 导入模型定义
            from PDiffuse_train import PointDiffuse

            # 创建模型实例
            num_classes = 13  # S3DIS默认13类
            model = PointDiffuse(num_classes=num_classes)

            # 加载权重
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            print(f"Loaded real model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading real model: {str(e)}")
            # 加载失败时回退到伪模型
            return DummyModel(device)
    else:
        # 使用伪模型
        print(f"No model found at {model_path}, using dummy model")
        return DummyModel(device)


def process_point_cloud(model, file_path):
    start_time = time.time()

    try:
        # 加载点云数据
        data = np.load(file_path)
        points = data[:, :3].astype(np.float32)

        # 如果数据包含颜色，则使用，否则创建默认颜色
        if data.shape[1] >= 6:
            colors = data[:, 3:6].astype(np.float32) / 255.0
        else:
            colors = np.ones((len(points), 3), dtype=np.float32) * 0.5  # 灰色

        features = np.concatenate([points, colors], axis=1)

        # 转换为tensor
        points_tensor = torch.from_numpy(points).unsqueeze(0).float()
        features_tensor = torch.from_numpy(features).unsqueeze(0).float()

        # 使用模型进行预测
        predictions, _ = model.predict(points_tensor, features_tensor)

        # 生成可视化
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 随机采样部分点用于可视化
        if len(points) > 10000:
            idx = np.random.choice(len(points), 10000, replace=False)
            sample_points = points[idx]
            sample_predictions = predictions[idx]
        else:
            sample_points = points
            sample_predictions = predictions

        ax.scatter(
            sample_points[:, 0],
            sample_points[:, 1],
            sample_points[:, 2],
            c=sample_predictions,
            cmap='tab20',
            s=1
        )
        ax.set_title('Point Cloud Segmentation')

        # 保存到内存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        # 类别分布统计
        unique, counts = np.unique(predictions, return_counts=True)
        class_distribution = {str(cls): int(count) for cls, count in zip(unique, counts)}

        processing_time = time.time() - start_time

        return {
            'visualization': img,
            'class_distribution': class_distribution,
            'points_count': len(points),
            'processing_time': processing_time
        }

    except Exception as e:
        # 生成错误可视化
        fig = plt.figure(figsize=(10, 8))
        plt.text(0.1, 0.5, f"Error: {str(e)}", fontsize=12)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return {
            'visualization': img,
            'class_distribution': {},
            'points_count': 0,
            'processing_time': 0,
            'error': str(e)
        }