import numpy as np
import cv2


def keep_largest_k_components(binary_mask: np.ndarray, k: int = 2, min_area: int = 50) -> np.ndarray:
    """
    保留最大的 k 个连通域，并去掉面积太小的碎点
    输入:
        binary_mask: 2D numpy array, 值为 0/1 或 False/True
    输出:
        cleaned_mask: 2D uint8 array, 值为 0/1
    """
    binary_mask = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # 收集所有前景连通域
    components = []
    for label_idx in range(1, num_labels):  # 0 是背景
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((label_idx, area))

    # 按面积从大到小排序
    components.sort(key=lambda x: x[1], reverse=True)

    # 只保留最大的 k 个
    keep_labels = set(label for label, _ in components[:k])

    cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for label_idx in keep_labels:
        cleaned_mask[labels == label_idx] = 1

    return cleaned_mask


def postprocess_prediction(
    pred_mask: np.ndarray,
    threshold: float = 0.5,
    keep_k: int = 2,
    min_area: int = 50
) -> np.ndarray:
    """
    对预测结果做后处理
    pred_mask:
        可以是概率图(0~1)，也可以是二值图
    """
    # 二值化
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    # 保留最大两个连通域
    cleaned_mask = keep_largest_k_components(binary_mask, k=keep_k, min_area=min_area)

    return cleaned_mask