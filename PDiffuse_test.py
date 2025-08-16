import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Import model components from training script
from PDiffuse_train import PointDiffuse, S3DISDataset, ScanNetDataset


class PointDiffuseInference:
    def __init__(self, model, device, num_diffusion_steps=20, sampling_method='ddim', eta=0.0):
        self.model = model.to(device)
        self.device = device
        self.num_diffusion_steps = num_diffusion_steps
        self.sampling_method = sampling_method.lower()
        self.eta = eta  # DDIM parameter, 0.0 for deterministic sampling
        self.model.eval()

        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)

        # Log sampling method info
        if self.sampling_method == 'ddim':
            logging.info("Using DDIM sampling method")
            logging.info("DDIM provides ~20x speedup as mentioned in the paper")
        else:
            logging.info("Using DDPM sampling method")

    def _ensure_consistent_dtype(self, tensor, target_dtype):
        """确保张量数据类型一致"""
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

    @torch.no_grad()
    def reverse_diffusion_ddim(self, semantic_condition, position_condition, shape, batch_size_limit=1024):
        """DDIM reverse diffusion process for inference (as described in paper)"""
        batch_size, num_points, num_classes = shape

        # Process in chunks if number of points is too large
        if num_points > batch_size_limit:
            return self._reverse_diffusion_chunked(semantic_condition, position_condition, shape, batch_size_limit,
                                                   method='ddim')

        # Ensure consistent dtype
        target_dtype = semantic_condition.dtype

        # Start from pure noise
        x_t = torch.randn(batch_size, num_points, num_classes, dtype=target_dtype).to(self.device)

        # DDIM sampling schedule - skip steps for acceleration
        # For 20 steps from 1000 training steps, we sample every 50 steps
        total_training_steps = 1000
        skip = total_training_steps // self.num_diffusion_steps
        seq = range(0, total_training_steps, skip)
        seq = [int(s) for s in list(seq)]

        # Reverse the sequence for sampling
        seq_next = [-1] + list(seq[:-1])

        for i, (t_cur, t_next) in enumerate(tqdm(zip(reversed(seq), reversed(seq_next)),
                                                 desc="DDIM reverse diffusion",
                                                 total=len(seq))):
            t_tensor = torch.full((batch_size,), t_cur, dtype=torch.long).to(self.device)

            # Predict noise with mixed precision
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                noise_pred = self.model.denoising_net(x_t, semantic_condition, position_condition, t_tensor)

            # Ensure consistent dtype
            noise_pred = self._ensure_consistent_dtype(noise_pred, target_dtype)

            # DDIM sampling step
            alpha_t = self.model.alphas_cumprod[t_cur]

            if t_next >= 0:
                alpha_t_next = self.model.alphas_cumprod[t_next]
            else:
                alpha_t_next = torch.tensor(1.0).to(self.device)

            # Predict x_0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

            if t_next >= 0:
                # Direction to x_t
                sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
                sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

                # DDIM update rule
                dir_xt = sqrt_one_minus_alpha_t_next * noise_pred

                # Add stochasticity if eta > 0
                if self.eta > 0:
                    sigma_t = self.eta * torch.sqrt(
                        (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)
                    )
                    noise = torch.randn_like(x_t)
                    x_t = sqrt_alpha_t_next * x_0_pred + dir_xt + sigma_t * noise
                else:
                    # Deterministic DDIM
                    x_t = sqrt_alpha_t_next * x_0_pred + dir_xt
            else:
                # Final step
                x_t = x_0_pred

            # Clear cache periodically
            if i % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        return x_t

    @torch.no_grad()
    def reverse_diffusion_ddpm(self, semantic_condition, position_condition, shape, batch_size_limit=1024):
        """Original DDPM reverse diffusion process"""
        batch_size, num_points, num_classes = shape

        # Process in chunks if number of points is too large
        if num_points > batch_size_limit:
            return self._reverse_diffusion_chunked(semantic_condition, position_condition, shape, batch_size_limit,
                                                   method='ddpm')

        # Ensure consistent dtype
        target_dtype = semantic_condition.dtype

        # Start from pure noise
        x_t = torch.randn(batch_size, num_points, num_classes, dtype=target_dtype).to(self.device)

        # Reverse diffusion steps
        for i in tqdm(reversed(range(self.num_diffusion_steps)), desc="DDPM reverse diffusion"):
            t = torch.full((batch_size,), i, dtype=torch.long).to(self.device)

            # Predict noise
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                noise_pred = self.model.denoising_net(x_t, semantic_condition, position_condition, t)

            # Ensure consistent dtype
            noise_pred = self._ensure_consistent_dtype(noise_pred, target_dtype)

            # Compute previous step
            if i > 0:
                alpha_t = self.model.alphas[i]
                alpha_cumprod_t = self.model.alphas_cumprod[i]
                alpha_cumprod_t_prev = self.model.alphas_cumprod[i - 1]
                beta_t = self.model.betas[i]

                # DDPM reverse step
                x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

                # Direction to x_t
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred

                x_t = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + dir_xt

                if i > 1:
                    # Add noise
                    noise = torch.randn_like(x_t)
                    x_t = x_t + torch.sqrt(beta_t) * noise
            else:
                # Final step
                alpha_cumprod_t = self.model.alphas_cumprod[i]
                x_t = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

            # Clear cache periodically
            if i % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        return x_t

    @torch.no_grad()
    def reverse_diffusion(self, semantic_condition, position_condition, shape, batch_size_limit=1024):
        """选择采样方法"""
        if self.sampling_method == 'ddim':
            return self.reverse_diffusion_ddim(semantic_condition, position_condition, shape, batch_size_limit)
        else:
            return self.reverse_diffusion_ddpm(semantic_condition, position_condition, shape, batch_size_limit)

    @torch.no_grad()
    def _reverse_diffusion_chunked(self, semantic_condition, position_condition, shape, chunk_size, method='ddim'):
        """Process large point clouds in chunks"""
        batch_size, num_points, num_classes = shape
        results = []

        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)
            chunk_points = end_idx - start_idx

            # Extract chunks
            sem_cond_chunk = semantic_condition[:, start_idx:end_idx]
            pos_cond_chunk = position_condition[:, start_idx:end_idx]

            # Process chunk
            if method == 'ddim':
                chunk_result = self.reverse_diffusion_ddim(
                    sem_cond_chunk, pos_cond_chunk,
                    (batch_size, chunk_points, num_classes)
                )
            else:
                chunk_result = self.reverse_diffusion_ddpm(
                    sem_cond_chunk, pos_cond_chunk,
                    (batch_size, chunk_points, num_classes)
                )

            results.append(chunk_result)

            # Clear memory
            del sem_cond_chunk, pos_cond_chunk, chunk_result
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return torch.cat(results, dim=1)

    def predict(self, points, features, max_points_per_batch=8192):
        """Memory-optimized prediction with point cloud chunking and dtype consistency"""
        original_shape = points.shape
        batch_size, num_points, _ = original_shape

        # 确保输入数据类型一致
        if points.dtype != features.dtype:
            # 统一使用 float32 以避免混合精度问题
            points = points.float()
            features = features.float()

        # If point cloud is too large, process in chunks
        if num_points > max_points_per_batch:
            return self._predict_chunked(points, features, max_points_per_batch)

        # Clear cache before processing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        try:
            # Get conditions - 在这里不使用半精度以避免dtype不匹配
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                semantic_condition, position_condition = self.model(points, features)

            # Ensure consistent dtype for conditions
            target_dtype = points.dtype
            semantic_condition = self._ensure_consistent_dtype(semantic_condition, target_dtype)
            position_condition = self._ensure_consistent_dtype(position_condition, target_dtype)

            # Reverse diffusion to generate labels
            label_logits = self.reverse_diffusion(
                semantic_condition, position_condition,
                (batch_size, num_points, self.model.num_classes)
            )

            # Convert to class predictions
            predictions = torch.argmax(label_logits, dim=-1)

            return predictions, label_logits

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Fallback to even smaller chunks
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                return self._predict_chunked(points, features, max_points_per_batch // 2)
            else:
                raise e

    @torch.no_grad()
    def _predict_chunked(self, points, features, chunk_size):
        """Process large point clouds in smaller chunks with dtype consistency"""
        batch_size, num_points, _ = points.shape
        all_predictions = []
        all_logits = []

        # 确保数据类型一致
        target_dtype = points.dtype

        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)

            # Extract chunk
            points_chunk = points[:, start_idx:end_idx]
            features_chunk = features[:, start_idx:end_idx]

            # 确保chunk数据类型一致
            points_chunk = self._ensure_consistent_dtype(points_chunk, target_dtype)
            features_chunk = self._ensure_consistent_dtype(features_chunk, target_dtype)

            # Process chunk
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                semantic_condition, position_condition = self.model(points_chunk, features_chunk)

            # 确保条件数据类型一致
            semantic_condition = self._ensure_consistent_dtype(semantic_condition, target_dtype)
            position_condition = self._ensure_consistent_dtype(position_condition, target_dtype)

            # Reverse diffusion for chunk
            chunk_logits = self.reverse_diffusion(
                semantic_condition, position_condition,
                (batch_size, end_idx - start_idx, self.model.num_classes)
            )

            chunk_predictions = torch.argmax(chunk_logits, dim=-1)

            all_predictions.append(chunk_predictions)
            all_logits.append(chunk_logits)

            # Clean up
            del points_chunk, features_chunk, semantic_condition, position_condition
            del chunk_logits, chunk_predictions
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Concatenate results
        predictions = torch.cat(all_predictions, dim=1)
        logits = torch.cat(all_logits, dim=1)

        return predictions, logits


def calculate_metrics(predictions, targets, num_classes):
    """Calculate segmentation metrics"""
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Overall accuracy
    overall_acc = accuracy_score(targets_flat, predictions_flat)

    # Per-class IoU and accuracy
    class_iou = []
    class_acc = []

    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((predictions_flat == i) & (targets_flat == i))
        fp = np.sum((predictions_flat == i) & (targets_flat != i))
        fn = np.sum((predictions_flat != i) & (targets_flat == i))

        # IoU
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 0.0
        class_iou.append(iou)

        # Class accuracy
        class_total = np.sum(targets_flat == i)
        if class_total > 0:
            acc = tp / class_total
        else:
            acc = 0.0
        class_acc.append(acc)

    mean_iou = np.mean(class_iou)
    mean_acc = np.mean(class_acc)

    return {
        'overall_accuracy': overall_acc,
        'mean_iou': mean_iou,
        'mean_accuracy': mean_acc,
        'class_iou': class_iou,
        'class_accuracy': class_acc
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_predictions(points, predictions, targets, save_path, max_points=10000):
    """Visualize point cloud predictions"""
    # Subsample for visualization
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        predictions = predictions[idx]
        targets = targets[idx]

    fig = plt.figure(figsize=(15, 5))

    # Ground truth
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=targets, cmap='tab20', s=1)
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Predictions
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=predictions, cmap='tab20', s=1)
    ax2.set_title('Predictions')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Error map
    ax3 = fig.add_subplot(133, projection='3d')
    errors = (predictions != targets).astype(int)
    scatter3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=errors, cmap='coolwarm', s=1)
    ax3.set_title('Errors (Red=Wrong, Blue=Correct)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(model_path, data_root, dataset_type, device,
                   test_area=5, num_diffusion_steps=20, save_dir='./results',
                   sampling_method='ddim', eta=0.0):
    """Memory-optimized model evaluation with DDIM/DDPM support"""

    # Set memory optimization settings
    if device.type == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']

    # Determine number of classes
    if dataset_type == 's3dis':
        num_classes = 13
        class_names = ['ceiling', 'floor', 'wall', 'beam', 'column',
                       'window', 'door', 'table', 'chair', 'sofa',
                       'bookcase', 'board', 'clutter']
    else:  # scannet
        num_classes = 20
        class_names = [f'class_{i}' for i in range(num_classes)]  # Simplified

    # Create model
    model = PointDiffuse(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 不转换为半精度以避免dtype问题
    # model = model.half()  # 注释掉这行

    # Create inference wrapper
    inference = PointDiffuseInference(model, device, num_diffusion_steps,
                                      sampling_method, eta)

    # Create dataset
    if dataset_type == 's3dis':
        test_dataset = S3DISDataset(data_root, split='val', test_area=test_area)
    else:  # scannet
        test_dataset = ScanNetDataset(data_root, split='val')

    # Use smaller batch size and more workers
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    # Evaluation
    all_predictions = []
    all_targets = []
    all_points = []

    os.makedirs(save_dir, exist_ok=True)

    logging.info("Starting evaluation...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Clear cache before each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            points = batch['points'].to(device, non_blocking=True)
            features = batch['features'].to(device, non_blocking=True)
            targets = batch['labels']

            try:
                # Predict
                predictions, _ = inference.predict(points, features)

                # Move to CPU for metrics calculation
                points_cpu = points.cpu().numpy()
                predictions_cpu = predictions.cpu().numpy()
                targets_cpu = targets.numpy()

                all_predictions.append(predictions_cpu)
                all_targets.append(targets_cpu)
                all_points.append(points_cpu)

                # Save some visualizations
                if i < 5:  # Save first 5 scenes
                    vis_path = os.path.join(save_dir, f'scene_{i}_visualization.png')
                    visualize_predictions(points_cpu[0], predictions_cpu[0],
                                          targets_cpu[0], vis_path)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.warning(f"OOM error on scene {i}, skipping...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_points = np.concatenate(all_points, axis=0)

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets, num_classes)

    # Log results
    logging.info("=" * 50)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 50)
    logging.info(f"Sampling Method: {sampling_method.upper()}")
    logging.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    logging.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
    logging.info(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")

    logging.info("\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(class_names, metrics['class_iou'])):
        logging.info(f"{name}: {iou:.4f}")

    logging.info("\nPer-class Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, metrics['class_accuracy'])):
        logging.info(f"{name}: {acc:.4f}")

    # Save confusion matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_targets, all_predictions, class_names, cm_path)

    # Save metrics to file
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Sampling Method: {sampling_method.upper()}\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
        f.write(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}\n")

        f.write("\nPer-class IoU:\n")
        for name, iou in zip(class_names, metrics['class_iou']):
            f.write(f"{name}: {iou:.4f}\n")

        f.write("\nPer-class Accuracy:\n")
        for name, acc in zip(class_names, metrics['class_accuracy']):
            f.write(f"{name}: {acc:.4f}\n")

    return metrics


def inference_single_scene(model_path, points_path, device,
                           num_diffusion_steps=20, save_path=None,
                           sampling_method='ddim', eta=0.0):
    """Memory-optimized inference on a single point cloud scene with DDIM support"""

    # Set memory optimization settings
    if device.type == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']

    # Determine dataset type and classes
    if hasattr(args, 'dataset') and args.dataset == 'scannet':
        num_classes = 20
        class_names = [f'class_{i}' for i in range(num_classes)]
    else:
        num_classes = 13
        class_names = ['ceiling', 'floor', 'wall', 'beam', 'column',
                       'window', 'door', 'table', 'chair', 'sofa',
                       'bookcase', 'board', 'clutter']

    # Create model
    model = PointDiffuse(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 不转换为半精度以避免dtype问题
    # model = model.half()  # 注释掉这行

    # Create inference wrapper
    inference = PointDiffuseInference(model, device, num_diffusion_steps,
                                      sampling_method, eta)

    # Load point cloud
    if points_path.endswith('.npy'):
        data = np.load(points_path)
        if data.shape[1] >= 6:  # xyz + rgb + maybe labels
            points = data[:, :3].astype(np.float32)
            colors = data[:, 3:6].astype(np.float32) / 255.0
        else:
            points = data[:, :3].astype(np.float32)
            colors = np.ones((len(points), 3), dtype=np.float32)  # Default white
    else:
        raise ValueError("Only .npy files are supported")

    # Prepare input
    features = np.concatenate([points, colors], axis=1)
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

    # Inference
    if sampling_method == 'ddim':
        logging.info("Running inference with DDIM sampling...")
    else:
        logging.info("Running inference with DDPM sampling...")

    # Clear cache before inference
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    with torch.no_grad():
        predictions, logits = inference.predict(points_tensor, features_tensor)

    predictions_np = predictions.cpu().numpy()[0]
    logits_np = logits.cpu().numpy()[0]

    # Save results
    if save_path:
        result_data = {
            'points': points,
            'colors': colors * 255,  # Convert back to 0-255
            'predictions': predictions_np,
            'logits': logits_np,
            'class_names': class_names,
            'sampling_method': sampling_method
        }
        np.save(save_path, result_data)

        # Create visualization
        vis_path = save_path.replace('.npy', '_visualization.png')
        fig = plt.figure(figsize=(10, 5))

        # Original point cloud
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=colors, s=1)
        ax1.set_title('Original Point Cloud')

        # Segmentation result
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=predictions_np, cmap='tab20', s=1)
        ax2.set_title(f'Segmentation Result ({sampling_method.upper()})')

        plt.tight_layout()
        plt.savefig(vis_path, dpi=150)
        plt.close()

        logging.info(f"Results saved to {save_path}")
        logging.info(f"Visualization saved to {vis_path}")

    return predictions_np, logits_np


def main():
    parser = argparse.ArgumentParser(description='PointDiffuse Inference with DDIM/DDPM Support')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'single'],
                        default='evaluate', help='Inference mode')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str,
                        help='Root directory of dataset (for evaluate mode)')
    parser.add_argument('--points_path', type=str,
                        help='Path to single point cloud file (for single mode)')
    parser.add_argument('--dataset', type=str, choices=['s3dis', 'scannet'],
                        default='s3dis', help='Dataset type')
    parser.add_argument('--test_area', type=int, default=5,
                        help='Test area for S3DIS')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of diffusion steps')
    parser.add_argument('--save_dir', type=str, default='./inference_results',
                        help='Directory to save results')
    parser.add_argument('--save_path', type=str,
                        help='Path to save single scene results')
    parser.add_argument('--max_points', type=int, default=8192,
                        help='Maximum points per batch to avoid OOM')
    parser.add_argument('--sampling_method', type=str, choices=['ddim', 'ddpm'],
                        default='ddim', help='Sampling method (DDIM for speed as in paper, DDPM for original)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0.0 for deterministic, >0 for stochastic)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Set memory optimization environment variables
    if device.type == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Enable memory optimization
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    if args.mode == 'evaluate':
        if not args.data_root:
            raise ValueError("data_root is required for evaluate mode")

        os.makedirs(args.save_dir, exist_ok=True)

        # Setup logging to file
        log_handler = logging.FileHandler(os.path.join(args.save_dir, 'evaluation.log'))
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)

        # Run evaluation
        metrics = evaluate_model(
            model_path=args.model_path,
            data_root=args.data_root,
            dataset_type=args.dataset,
            device=device,
            test_area=args.test_area,
            num_diffusion_steps=args.num_steps,
            save_dir=args.save_dir,
            sampling_method=args.sampling_method,
            eta=args.eta
        )

        logging.info("Evaluation completed successfully!")

    elif args.mode == 'single':
        if not args.points_path:
            raise ValueError("points_path is required for single mode")

        predictions, logits = inference_single_scene(
            model_path=args.model_path,
            points_path=args.points_path,
            device=device,
            num_diffusion_steps=args.num_steps,
            save_path=args.save_path,
            sampling_method=args.sampling_method,
            eta=args.eta
        )

        logging.info("Single scene inference completed!")
        logging.info(f"Predicted classes: {np.unique(predictions)}")
        logging.info(f"Used sampling method: {args.sampling_method.upper()}")


if __name__ == '__main__':
    main()