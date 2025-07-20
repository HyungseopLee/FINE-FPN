import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

from pycocotools.coco import COCO
from PIL import Image
import random
import os    

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import seaborn as sns


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        sys.stdout.flush()

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import random
import os
from pycocotools.coco import COCO

# 91
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
    'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'streetsign', 'stopsign',
    'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite',
    'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle',
    'plate', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair',
    'sofa', 'pottedplant', 'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door',
    'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddybear',
    'hairdrier', 'toothbrush', 'hairbrush'
]
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_class_color(class_id):
    np.random.seed(class_id)
    return np.random.rand(3,)

def visualize(model, image_path, device, is_baseline=False):
    # 이미지 로딩
    image = Image.open(image_path).convert("RGB")
    
    # 이미지 전처리
    image = image.resize((640, 640))  # Resize to 640x640
    transform = T.Compose([
        T.ToTensor(),  # Mask R-CNN expects tensors in range [0, 1]
    ])
    image_tensor = transform(image).to(device)
    model.eval()

    with torch.no_grad():
        output = model([image_tensor])[0]  # 배치 크기 1

    # 시각화용 PIL 이미지로 변환
    image_np = np.array(image)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_np)

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()
    masks = output.get("masks")

    threshold = 0.5
    for i in range(len(boxes)):
        if scores[i] < threshold:
            continue

        label = labels[i].item()
        score = scores[i].item()

        # Check if label is valid
        if label >= len(COCO_CLASSES):
            print(f"Warning: Invalid class ID {label} for image {image_path}. Skipping.")
            continue

        # 클래스 이름
        class_name = COCO_CLASSES[label]

        # 고유 색상 생성 (클래스 인덱스를 기반으로)
        color = get_class_color(label)

        box = boxes[i].numpy()
        # color: tuple of floats (e.g., (0.2, 0.5, 0.8))

        # Bounding Box
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                linewidth=5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            box[0],
            box[1],
            f"{class_name}: {score:.2f}",
            color="white",
            fontsize=20,
            fontweight='bold',
            bbox=dict(
                facecolor=color,
                edgecolor='black',   # 테두리
                boxstyle='round,pad=0.3',  # 박스 스타일과 패딩
                alpha=0.9,
                linewidth=1.5
            )
        )

        # Mask
        mask = masks[i, 0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(image_np, dtype=np.uint8)

        r, g, b = [int(c * 255) for c in color]
        colored_mask[:, :, 0] = mask * r
        colored_mask[:, :, 1] = mask * g
        colored_mask[:, :, 2] = mask * b

        image_np = np.where(mask[..., None], 0.5 * image_np + 0.5 * colored_mask, image_np).astype(np.uint8)
        ax.imshow(image_np)


    # 이미지를 파일로 저장 (figure는 띄우지 않음)
    
    ax.axis("off")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 파일명만 추출
    if is_baseline:
        save_path = os.path.join(SAVE_DIR, f"{base_name}_baseline.png")
    else:
        save_path = os.path.join(SAVE_DIR, f"{base_name}_ours.png")
        
    plt.savefig(save_path, bbox_inches="tight", dpi=1500)
    plt.close(fig)  # figure 닫기
    print(f"Saved visualization to {save_path}")
    
def visualize_gt(coco, image_path, img_id, device):
    # 이미지 로딩
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()
    image = image.resize((640, 640))  # 모델과 동일 크기로 resize
    image_np = np.array(image)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_np)

    for ann in anns:
        bbox = ann["bbox"]  # [x, y, width, height]
        label = ann["category_id"]
        class_name = COCO_CLASSES[label]  # COCO 클래스 라벨에 맞는 이름
        color = get_class_color(label)

        # Calculate scaling factors
        scale_x = 640 / original_image.width
        scale_y = 640 / original_image.height
        
        # Adjust bounding box based on image scaling
        bbox = [coord * scale_x if i % 2 == 0 else coord * scale_y for i, coord in enumerate(bbox)]

        # Draw bounding box
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            box[0],
            box[1],
            f"{class_name}: {score:.2f}",
            color="white",
            fontsize=20,
            fontweight='bold',
            bbox=dict(
                facecolor=color,
                edgecolor='black',   # 테두리
                boxstyle='round,pad=0.3',  # 박스 스타일과 패딩
                alpha=0.9,
                linewidth=1.5
            )
        )
    
    
        if "segmentation" in ann:
            mask = coco.annToMask(ann)
            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = mask.resize((640, 640), Image.NEAREST)  # Resize
            mask = np.array(mask)
            colored_mask = np.zeros_like(image_np, dtype=np.uint8)
            colored_mask[:, :, 0] = mask * color[0] * 255
            colored_mask[:, :, 1] = mask * color[1] * 255
            colored_mask[:, :, 2] = mask * color[2] * 255
            # 마스크를 원본 이미지와 합성하여 적용
            image_np = np.where(mask[..., None], 0.5 * image_np + 0.5 * colored_mask, image_np).astype(np.uint8)
            ax.imshow(image_np)

    ax.axis("off")
    
    # 이미지를 파일로 저장 (figure는 띄우지 않음)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(SAVE_DIR, f"{base_name}_gt.png")
    
    plt.savefig(save_path, bbox_inches="tight", dpi=1500)
    plt.close(fig)  # figure 닫기
    print(f"Saved GT visualization to {save_path}")


def visualize_coco_samples(model, coco_ann_file, image_root, device, is_baseline=False, num_samples=30, seed=723):
    # COCO annotation 불러오기
    coco = COCO(coco_ann_file)

    # # 모든 이미지 ID 중에서 무작위로 샘플링 (매번 동일한 이미지 선택을 위해 시드 설정)
    # img_ids = coco.getImgIds()
    # random.seed(seed)  # 고정된 시드로 샘플링
    # sampled_ids = random.sample(img_ids, num_samples)
    
    sampled_ids = [
        311180,
        89697,
        60449,
        241326,
        466835
    ]
    
    for img_id in sampled_ids:
        
        print(f"Processing image ID: {img_id}")
        
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_root, img_info['file_name'])

        visualize(model, img_path, device=device, is_baseline=is_baseline)
        # visualize_gt(coco, img_path, img_id, device=device)



from matplotlib.colors import LinearSegmentedColormap
# light_reds = LinearSegmentedColormap.from_list("light_reds", ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d"])        

def visualize_cosine_similarity_of_hierarchical_features(
    aligned_hook: dict[str, dict[str, torch.Tensor]],
    misaligned_hook: dict[str, dict[str, torch.Tensor]],
    img_id: int = None,
    level_names: list[str] = None,
    is_baseline: bool = False,
    save_dir: str = "./semantic_cos_sim"
):
    if level_names is None:
        original_levels = sorted([k for k in hooks.keys() if k != '0'], reverse=True)
        level_names = [str(int(k) + 1) for k in original_levels]

    pooled_feats = []
    for level in sorted(hooks.keys(), reverse=True):
        if level == '0':
            continue
        feat = hooks[level]['inner_lateral']
        pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        pooled_feats.append(pooled)

    feats = torch.cat(pooled_feats, dim=0)
    feats = F.normalize(feats, p=2, dim=1)
    sim_matrix = torch.mm(feats, feats.t()).cpu().numpy()

    # Plotting
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        sim_matrix,
        xticklabels=level_names,
        yticklabels=level_names,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1.0,
        square=True,
        linewidths=0.2,
        cbar_kws={"shrink": 0.85, "label": "Cosine Similarity"},
        annot_kws={"size": 10, "color": "black"}
    )

    plt.title(f"Cosine Similarity between FPN Levels (Image ID: {img_id})", fontsize=12)
    plt.xlabel("FPN Level", fontsize=10)
    plt.ylabel("FPN Level", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    
    if is_baseline:
        save_path = os.path.join(save_dir, f"cosine_similarity_img{img_id}_baseline.png")
    else:
        save_path = os.path.join(save_dir, f"cosine_similarity_img{img_id}_ours.png")
        
    plt.savefig(save_path, dpi=300, transparent=True)
    print(f"✅ Saved cosine similarity heatmap: {save_path}")
    plt.close()
    
   

    
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

def visualize_tSNE_of_aligned_vs_misaligned_features(
    aligned_hooks: dict[str, dict[str, torch.Tensor]],
    misaligned_hooks: dict[str, dict[str, torch.Tensor]],
    img_id: int = None,
    level_names: list[str] = None,
    is_baseline: bool = False,
    save_dir: str = "./semantic_tSNE"
):
    os.makedirs(save_dir, exist_ok=True)

    if level_names is None:
        original_levels = sorted([k for k in aligned_hooks.keys() if k != '0'], reverse=True)
        level_names = [str(int(k) + 1) for k in original_levels]

    all_features = []
    all_labels = []
    all_styles = []

    for idx, level in enumerate(level_names):
        if level not in aligned_hooks or level not in misaligned_hooks:
            print(f"⚠️ Skipping level {level}: not in both hooks")
            continue

        for label_type, hooks, style in [("Aligned", aligned_hooks, "o"), ("Misaligned", misaligned_hooks, "X")]:
            feat = hooks[level]['inner_lateral'].squeeze(0)  # (C, H, W)
            feat = feat.flatten(1).T  # (H*W, C)

            # Optional: Subsample for speed
            if feat.shape[0] > 1000:
                idxs = torch.randperm(feat.shape[0])[:1000]
                feat = feat[idxs]

            all_features.append(feat.cpu().numpy())
            all_labels.extend([f"Level {level} ({label_type})"] * feat.shape[0])
            all_styles.extend([style] * feat.shape[0])

    all_features = np.concatenate(all_features, axis=0)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(all_labels))
    palette = sns.color_palette("hsv", len(unique_labels) // 2)

    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(all_labels) if l == label]
        style = all_styles[indices[0]]
        color = palette[i // 2]
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=label,
            s=20,
            c=[color],
            marker=style,
            alpha=0.7,
            edgecolors='w'
        )

    plt.title(f"t-SNE: Aligned vs Misaligned Features (img{img_id})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(fontsize=10, title="FPN Level + Type", loc='best')
    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"tsne_img{img_id}_{'baseline' if is_baseline else 'ours'}.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved aligned vs misaligned t-SNE plot: {save_path}")
    return tsne_results, save_path


def compare_aligned_misaligned_similarity(
    aligned_hook: dict,
    misaligned_hook: dict,
    img_id: int,
    save_dir: str = "./aligned_vs_misaligned"
):
    """Aligned와 Misaligned feature의 코사인 유사도 비교"""
    
    def compute_similarity_matrix(hooks):
        features = []
        original_levels = sorted([k for k in hooks.keys()], reverse=True)
        
        for level in original_levels:
            feat = hooks[level]['inner_lateral']  # [1, C, H, W]
            feat = feat.squeeze(0).flatten(1)  # [C, H*W]
            features.append(feat)
        
        # 최소 공간 차원으로 정렬
        min_spatial = min(feat.size(1) for feat in features)
        features_aligned = [feat[:, :min_spatial] for feat in features]
        features_normalized = [F.normalize(feat, p=2, dim=0) for feat in features_aligned]
        
        # 유사도 매트릭스 계산
        num_levels = len(features_normalized)
        sim_matrix = np.zeros((num_levels, num_levels))
        
        for i in range(num_levels):
            for j in range(num_levels):
                sim_per_loc = F.cosine_similarity(
                    features_normalized[i], 
                    features_normalized[j], 
                    dim=0
                )
                sim_matrix[i, j] = sim_per_loc.mean().item()
        
        return sim_matrix
    
    # 각각의 유사도 매트릭스 계산
    aligned_sim = compute_similarity_matrix(aligned_hook)
    misaligned_sim = compute_similarity_matrix(misaligned_hook)
    
    # 레벨 이름 생성
    original_levels = sorted([k for k in aligned_hook.keys() if k != '0'], reverse=True)
    level_names = [str(int(k) + 1) for k in original_levels]
    
    # 나란히 비교 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Misaligned feature 유사도  
    sns.heatmap(misaligned_sim,
                xticklabels=level_names, yticklabels=level_names, 
                annot=True, fmt=".2f", cmap="Reds",
                vmin=0, vmax=1.0, ax=ax2, square=True)
    ax2.set_title("Misaligned Features\nCosine Similarity")
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"similarity_comparison_img{img_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return aligned_sim, misaligned_sim, save_path


def get_multi_scale_logits(model, hooks, device):
    """
    각 FPN level의 feature에서 얻은 로짓(logits)을 반환

    Args:
        model: Mask R-CNN 모델
        hooks: backbone.fpn.aligned_features (Dict[str, Dict[str, Tensor]])

    Returns:
        Dict[str, Tensor]: key는 level 이름(str), value는 [B, num_classes] shape의 logits
    """
    logits_dict = {}
    box_head = model.roi_heads.box_head
    cls_score = model.roi_heads.box_predictor.cls_score

    for level, feat_dict in hooks.items():
        feat = feat_dict['inner_lateral']  # 💡 핵심 수정
        B, C, H, W = feat.shape
        
        # print(f"Processing level {level}: {B}x{C}x{H}x{W}")
        # print(f"Feature shape: {feat.shape}")
        
        # make 7 x 7
        if H != 7 or W != 7:
            feat = F.adaptive_avg_pool2d(feat, (7, 7))
        # print(f"Resized feature shape: {feat.shape}")

        # Flatten: [B, C, H, W] → [B, C * H * W]
        flattened_feat = feat.view(B, -1).to(device)

        with torch.no_grad():
            x = box_head(flattened_feat)  # [B, 1024]
            logits = cls_score(x)         # [B, num_classes]

        logits_dict[level] = logits
        
        # print(f"Logits for level {level}: {logits.shape}")

    return logits_dict

import torch
import torch.nn.functional as F

def get_similarity_with_multi_scale_logits(multi_scale_logits: dict[str, torch.Tensor],
                                           method: str = 'KL'):
    """
    return the similarity between logits of adjacent FPN levels

    Args:
        multi_scale_logits (Dict[str, Tensor]): 
            key: level 이름(str), value: [B, num_classes] 로짓 tensor
        method (str): 'cosine' 또는 'euclidean'

    Returns:
        Dict[Tuple[str,str], Tensor]: 
            (lev_i, lev_j) 쌍마다 [B] shape의 similarity tensor
    """
    # 레벨 이름을 숫자 기준 내림차순 정렬
    levels = sorted(multi_scale_logits.keys(), key=lambda x: int(x), reverse=True)
    sims = {}

    for i in range(len(levels) - 1):
        l1, l2 = levels[i], levels[i+1]
        logits1 = multi_scale_logits[l1]  # [B, C]
        logits2 = multi_scale_logits[l2]

        if method == 'cosine':
            # 배치 내 각 샘플마다 cosine similarity
            sim = F.cosine_similarity(logits1, logits2, dim=1)  # [B]
        elif method == 'euclidean':
            # 배치 내 각 샘플마다 L2 거리
            sim = -torch.norm(logits1 - logits2, p=2, dim=1)  # 음수로 하면 유사도가 클수록 값이 크도록
        elif method == 'KL':
            prob1 = F.softmax(logits1, dim=1)
            prob2 = F.softmax(logits2, dim=1)
            sim = F.kl_div(prob1.log(), prob2, reduction='none').sum(dim=1)

        sims[(l1, l2)] = sim.cpu()  # CPU로 내보내기

    return sims

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import torch
import torchvision.transforms as T
import os
import json
from tqdm import tqdm
import numpy as np

def coco_val_fp_fn(model, device):
    """
    COCO validation에서 False Positive, False Negative를 계산하고 출력합니다.
    """
    coco_ann_file = "/media/data/coco/annotations/instances_val2017.json"
    coco_img_dir = "/media/data/coco/images/val2017"
    
    # COCO API 초기화
    coco = COCO(coco_ann_file)
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    img_ids = coco.getImgIds()
    results = []

    transform = T.Compose([T.ToTensor()])

    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_img_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)[0]  # single image output

        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > 0.5:
                result = {
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": [round(float(x), 2) for x in box.tolist()],
                    "score": float(score)
                }
                results.append(result)

    # 결과 저장
    results_file = "coco_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f)

    # 평가
    coco_dt = coco.loadRes(results_file)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    # FP/FN 계산
    fp, fn, tp = 0, 0, 0
    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue

        dt_matches = np.array(eval_img['dtMatches'][0])     # IoU=0.5
        dt_ignore = np.array(eval_img['dtIgnore'][0], dtype=bool)
        gt_ignore = np.array(eval_img['gtIgnore'], dtype=bool)

        # False Positive: 예측했지만 GT와 매칭되지 않음
        fp += int(((dt_matches == 0) & (~dt_ignore)).sum())

        # True Positive: 매칭된 예측
        tp += int(((dt_matches > 0) & (~dt_ignore)).sum())

        # False Negative: GT가 있었지만 매칭되지 않음
        fn += int((~gt_ignore).sum()) - int((dt_matches > 0).sum())

    print("=== COCO Evaluation Results ===")
    print(f"True Positive: {tp}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")