import torch
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import time
import traceback

# 设置matplotlib后端为Agg（无GUI）
import matplotlib

matplotlib.use('Agg')

# 添加训练代码路径到 sys.path
sys.path.append('/app')

# S3DIS类别名称
S3DIS_CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter'
]

# ScanNet类别名称（简化版）
SCANNET_CLASS_NAMES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair',
    'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'other'
]

# 尝试导入真实的模型定义
try:
    # 你需要将 PDiffuse_train.py 也复制到 Docker 容器中
    from PDiffuse_train import PointDiffuse

    REAL_MODEL_AVAILABLE = True
    print("✅ PDiffuse_train module imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: PDiffuse_train not found - {str(e)}")
    print("Will use dummy model for demonstration")
    REAL_MODEL_AVAILABLE = False

# 添加 PDiffuse_test.py 中的推理类
try:
    from PDiffuse_test import PointDiffuseInference

    INFERENCE_CLASS_AVAILABLE = True
    print("✅ PointDiffuseInference imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: PDiffuse_test not found - {str(e)}")
    print("Will use basic inference methods")
    INFERENCE_CLASS_AVAILABLE = False


class RealModelWrapper:
    """真实 PointDiffuse 模型的包装器"""

    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path

        print(f"🔄 Loading real PointDiffuse model from: {model_path}")
        print(f"🔄 Using device: {device}")

        try:
            # 加载检查点
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print("✅ Checkpoint loaded successfully")

            # 获取模型参数
            args = checkpoint.get('args', None)

            # 确定类别数量
            if args and hasattr(args, 'dataset'):
                if args.dataset == 'scannet':
                    self.num_classes = 20
                    self.class_names = SCANNET_CLASS_NAMES
                else:  # s3dis
                    self.num_classes = 13
                    self.class_names = S3DIS_CLASS_NAMES
            else:
                # 默认使用 S3DIS
                self.num_classes = 13
                self.class_names = S3DIS_CLASS_NAMES

            print(f"📊 Model classes: {self.num_classes}")
            print(f"📊 Dataset type: {getattr(args, 'dataset', 'unknown')}")

            # 创建模型实例
            self.model = PointDiffuse(
                num_classes=self.num_classes,
                in_channels=6,  # xyz + rgb
                channels=getattr(args, 'channels', 64),
                timesteps=getattr(args, 'timesteps', 1000)
            )

            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            print("✅ Model weights loaded and set to eval mode")

            # 创建推理包装器（使用 DDIM 加速）
            if INFERENCE_CLASS_AVAILABLE:
                self.inference = PointDiffuseInference(
                    model=self.model,
                    device=device,
                    num_diffusion_steps=20,  # 减少步数以加快推理
                    sampling_method='ddim',  # 使用 DDIM 加速
                    eta=0.0  # 确定性采样
                )
                print("✅ PointDiffuseInference wrapper created with DDIM sampling")
            else:
                self.inference = None
                print("⚠️ Using basic model sampling (slower)")

            # 计算模型参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"📈 Total model parameters: {total_params / 1e6:.2f}M")

            print("🎉 Real PointDiffuse model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading real model: {str(e)}")
            print("🔍 Full traceback:")
            traceback.print_exc()
            raise e

    def predict(self, points_tensor, features_tensor):
        """使用真实模型进行预测"""
        print(f"🔄 Starting real model inference...")
        print(f"📊 Input shape - Points: {points_tensor.shape}, Features: {features_tensor.shape}")

        start_time = time.time()

        try:
            with torch.no_grad():
                # 确保数据在正确的设备上
                points_tensor = points_tensor.to(self.device)
                features_tensor = features_tensor.to(self.device)

                if self.inference:
                    # 使用优化的推理类
                    print("🚀 Using optimized DDIM inference...")
                    predictions, logits = self.inference.predict(
                        points_tensor, features_tensor,
                        max_points_per_batch=4096  # 限制批次大小以节省内存
                    )
                    predictions = predictions.cpu().numpy()
                    logits = logits.cpu().numpy()

                else:
                    # 使用基本的模型采样
                    print("🐌 Using basic DDIM sampling...")
                    logits = self.model.ddim_sample(points_tensor, features_tensor, num_steps=20)
                    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                    logits = logits.cpu().numpy()

                inference_time = time.time() - start_time
                print(f"✅ Real model inference completed in {inference_time:.2f} seconds")
                print(f"📊 Output shape - Predictions: {predictions.shape}, Logits: {logits.shape}")

                # 检查预测结果的合理性
                unique_classes = np.unique(predictions[0] if len(predictions.shape) > 1 else predictions)
                print(f"🎯 Predicted classes: {unique_classes}")

                return (predictions[0] if len(predictions.shape) > 1 else predictions,
                        logits[0] if len(logits.shape) > 2 else logits)

        except Exception as e:
            print(f"❌ Error during real model inference: {str(e)}")
            print("🔍 Full traceback:")
            traceback.print_exc()
            raise e


class DummyModel:
    """伪模型类，用于在没有真实模型时提供基本功能"""

    def __init__(self, device):
        self.device = device
        self.num_classes = 13  # S3DIS默认13类
        self.class_names = S3DIS_CLASS_NAMES
        print("🎭 Initialized Dummy Model for demonstration")

    def predict(self, points_tensor, features_tensor):
        """生成伪预测结果"""
        print("🎭 Using Dummy Model (demonstration only)")

        # 生成伪预测结果
        points = points_tensor.cpu().numpy()[0]
        num_points = points.shape[0]

        print(f"📊 Processing {num_points} points with dummy model")

        # 基于点的位置生成更合理的预测结果
        predictions = self._generate_realistic_predictions(points)

        # 生成伪logits
        logits = np.random.rand(num_points, self.num_classes)

        # 让logits与predictions一致
        for i in range(num_points):
            logits[i] = np.random.rand(self.num_classes) * 0.3  # 低置信度
            logits[i, predictions[i]] = np.random.rand() * 0.7 + 0.3  # 高置信度给预测类别

        print("✅ Dummy model prediction completed")
        return predictions, logits

    def _generate_realistic_predictions(self, points):
        """基于点云位置生成更合理的分割结果"""
        num_points = len(points)
        predictions = np.zeros(num_points, dtype=np.int32)

        # 获取点云的边界
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        ranges = max_coords - min_coords

        print(f"📊 Point cloud bounds: min={min_coords}, max={max_coords}")

        for i, point in enumerate(points):
            x, y, z = point

            # 归一化坐标 [0, 1]
            if ranges[2] > 0:  # Z 方向有变化
                relative_height = (z - min_coords[2]) / ranges[2]
            else:
                relative_height = 0.5

            if ranges[0] > 0:  # X 方向有变化
                relative_x = (x - min_coords[0]) / ranges[0]
            else:
                relative_x = 0.5

            if ranges[1] > 0:  # Y 方向有变化
                relative_y = (y - min_coords[1]) / ranges[1]
            else:
                relative_y = 0.5

            # 基于高度的启发式规则
            if relative_height > 0.85:
                predictions[i] = 0  # ceiling
            elif relative_height < 0.15:
                predictions[i] = 1  # floor
            elif relative_height > 0.6:
                # 高处：墙壁、窗户、门、黑板
                if relative_x < 0.1 or relative_x > 0.9 or relative_y < 0.1 or relative_y > 0.9:
                    predictions[i] = 2  # wall (边界更可能是墙)
                else:
                    predictions[i] = np.random.choice([2, 5, 6, 11],
                                                      p=[0.4, 0.2, 0.2, 0.2])  # wall, window, door, board
            elif relative_height > 0.3:
                # 中等高度：桌子、椅子、沙发、书柜
                predictions[i] = np.random.choice([7, 8, 9, 10], p=[0.3, 0.3, 0.2, 0.2])  # table, chair, sofa, bookcase
            else:
                # 低处：墙壁基座、杂物
                predictions[i] = np.random.choice([2, 12], p=[0.6, 0.4])  # wall, clutter

        # 统计生成的类别分布
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"🎯 Dummy model generated classes: {dict(zip(unique, counts))}")

        return predictions


def load_model():
    """加载模型（真实或伪模型）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.getenv('MODEL_PATH', '/app/model/best_model.pth')

    print("=" * 60)
    print("🚀 PointDiffuse Model Loading")
    print("=" * 60)
    print(f"🔍 Looking for model at: {model_path}")
    print(f"🔍 Real model available: {REAL_MODEL_AVAILABLE}")
    print(f"🔍 Inference class available: {INFERENCE_CLASS_AVAILABLE}")
    print(f"🔍 Using device: {device}")
    print("-" * 60)

    # 检查是否可以加载真实模型
    if os.path.exists(model_path) and REAL_MODEL_AVAILABLE:
        try:
            # 尝试加载真实模型
            model = RealModelWrapper(model_path, device)
            print("🎉 Successfully loaded REAL PointDiffuse model!")
            print("📈 This will provide high-quality semantic segmentation")
            print("⏰ Note: Inference will take longer but results will be accurate")
            return model

        except Exception as e:
            print(f"❌ Error loading real model: {str(e)}")
            print("🔄 Falling back to dummy model...")
    else:
        # 显示为什么使用伪模型
        reasons = []
        if not os.path.exists(model_path):
            reasons.append(f"❌ No model file found at {model_path}")
        if not REAL_MODEL_AVAILABLE:
            reasons.append("❌ PDiffuse_train.py not available")

        for reason in reasons:
            print(reason)

    # 使用伪模型
    print("🎭 Using Dummy Model for demonstration")
    print("📝 This provides quick results but is NOT real AI inference")
    print("📝 To use real model: place model file and copy PDiffuse_train.py")
    print("=" * 60)
    return DummyModel(device)


def process_point_cloud(model, file_path):
    """处理点云文件"""
    print(f"🔄 Processing point cloud: {file_path}")
    start_time = time.time()

    try:
        # 加载点云数据
        print("📂 Loading point cloud data...")
        if file_path.endswith('.npz'):
            data_dict = np.load(file_path)
            # 尝试不同的键名
            if 'data' in data_dict:
                data = data_dict['data']
            elif 'points' in data_dict:
                data = data_dict['points']
            elif 'coords' in data_dict:
                data = data_dict['coords']
            else:
                # 使用第一个可用的键
                key = list(data_dict.keys())[0]
                data = data_dict[key]
                print(f"📂 Using key '{key}' from .npz file")
        else:
            data = np.load(file_path)

        print(f"📊 Loaded data shape: {data.shape}")

        # 确保数据是2D数组
        if data.ndim == 1:
            raise ValueError("Data must be 2D array with shape (N, features)")

        if data.shape[1] < 3:
            raise ValueError("Data must have at least 3 columns (x, y, z coordinates)")

        points = data[:, :3].astype(np.float32)

        # 检查点云是否为空
        if len(points) == 0:
            raise ValueError("Point cloud is empty")

        print(f"📊 Point cloud has {len(points)} points")

        # 处理颜色信息
        if data.shape[1] >= 6:
            colors = data[:, 3:6].astype(np.float32)
            # 归一化颜色值（如果需要）
            if colors.max() > 1.0:
                colors = colors / 255.0
            print("🎨 Using RGB colors from data")
        else:
            colors = np.ones((len(points), 3), dtype=np.float32) * 0.5  # 灰色
            print("🎨 Using default gray colors")

        # 数据预处理
        original_count = len(points)

        # 如果点数太多，进行采样
        max_points = 50000  # 降低最大点数以提高性能
        if len(points) > max_points:
            print(f"🔄 Sampling {max_points} points from {len(points)} points")
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        # 归一化点云坐标（重要！）
        print("🔄 Normalizing point cloud coordinates...")
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist

        # 准备特征
        features = np.concatenate([points, colors], axis=1)  # [N, 6] - xyz + rgb

        print(f"📊 Final processing: {len(points)} points, {features.shape[1]} features")

        # 转换为tensor
        device = model.device if hasattr(model, 'device') else 'cpu'
        points_tensor = torch.from_numpy(points).unsqueeze(0).float().to(device)
        features_tensor = torch.from_numpy(features).unsqueeze(0).float().to(device)

        print(f"📊 Tensor shapes - Points: {points_tensor.shape}, Features: {features_tensor.shape}")

        # 使用模型进行预测
        print("🧠 Running model inference...")
        with torch.no_grad():
            predictions, logits = model.predict(points_tensor, features_tensor)

        print(f"📊 Prediction results - Shape: {predictions.shape}")

        # 生成可视化
        print("🎨 Generating visualization...")
        fig = plt.figure(figsize=(15, 10))

        # 随机采样部分点用于可视化以提高性能
        viz_points = 8000
        if len(points) > viz_points:
            viz_idx = np.random.choice(len(points), viz_points, replace=False)
            sample_points = points[viz_idx]
            sample_predictions = predictions[viz_idx]
            sample_colors = colors[viz_idx]
        else:
            sample_points = points
            sample_predictions = predictions
            sample_colors = colors

        # 创建两个子图：原始点云和分割结果
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # 原始点云（使用真实颜色）
        ax1.scatter(
            sample_points[:, 0],
            sample_points[:, 1],
            sample_points[:, 2],
            c=sample_colors,
            s=2,
            alpha=0.8
        )
        ax1.set_title('Original Point Cloud', fontsize=14)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 分割结果（使用类别颜色）
        scatter = ax2.scatter(
            sample_points[:, 0],
            sample_points[:, 1],
            sample_points[:, 2],
            c=sample_predictions,
            cmap='tab20',
            s=2,
            alpha=0.8
        )
        ax2.set_title('Semantic Segmentation', fontsize=14)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8, aspect=20)
        cbar.set_label('Class ID')

        # 设置相同的视角
        for ax in [ax1, ax2]:
            ax.view_init(elev=20, azim=45)
            ax.set_box_aspect([1, 1, 0.8])

        plt.suptitle(f'PointDiffuse Results ({type(model).__name__})', fontsize=16)
        plt.tight_layout()

        # 保存到内存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        # 类别分布统计
        unique, counts = np.unique(predictions, return_counts=True)
        class_distribution = {}

        # 使用模型的类别名称
        class_names = getattr(model, 'class_names', S3DIS_CLASS_NAMES)

        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                class_name = class_names[cls]
            else:
                class_name = f'class_{cls}'
            class_distribution[class_name] = int(count)

        processing_time = time.time() - start_time

        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Class distribution: {class_distribution}")

        return {
            'visualization': img,
            'class_distribution': class_distribution,
            'points_count': len(points),
            'original_points_count': original_count,
            'processing_time': processing_time,
            'model_type': type(model).__name__
        }

    except Exception as e:
        print(f"❌ Error in process_point_cloud: {str(e)}")
        print("🔍 Full traceback:")
        traceback.print_exc()

        # 生成错误可视化
        fig, ax = plt.subplots(figsize=(12, 8))
        error_text = f"""
        Error processing point cloud:

        File: {os.path.basename(file_path)}
        Error: {str(e)}

        Please check:
        • File format (.npy or .npz)
        • Data shape (should be N×3 or N×6)
        • File is not corrupted

        Model type: {type(model).__name__}
        """

        ax.text(0.5, 0.5, error_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Point Cloud Processing Error", fontsize=16, pad=20)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return {
            'visualization': img,
            'class_distribution': {},
            'points_count': 0,
            'original_points_count': 0,
            'processing_time': 0,
            'model_type': type(model).__name__,
            'error': str(e)
        }