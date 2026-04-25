from pathlib import Path
import csv
import yaml
import nibabel as nib
import numpy as np
import cv2


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_case_list(txt_path: Path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_connected_component_areas(binary_mask):
    """
    返回所有前景连通域面积（不含背景）
    """
    binary_u8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_u8, connectivity=8
    )

    areas = []
    for label_idx in range(1, num_labels):  # 0 是背景
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        areas.append(area)

    areas.sort(reverse=True)
    return areas


def is_valid_mask_slice(
    mask_slice,
    min_positive_pixels=500,
    min_mask_ratio=0.002,
    margin=10,
    min_bbox_h=40,
    min_bbox_w=20,
    min_largest_cc_area=200,
    min_top2_cc_sum=500,
    min_fill_ratio=0.15,
    min_upper_pixels=50,
    min_lower_pixels=50
):
    """
    判断一个 slice 是否保留：
    1. mask 像素数不能太少
    2. mask 占比不能太小
    3. mask 不能太贴图像边缘
    4. bbox 尺寸不能太小
    5. 最大连通域面积不能太小
    6. 前两个最大连通域面积和不能太小
    7. 目标不能只是“很薄的弧线”（fill_ratio 过滤）
    8. 上下两部分都要有一定量的肺结构（single_side_structure 过滤）
    """
    h, w = mask_slice.shape
    total_pixels = h * w

    binary_mask = (mask_slice > 0)
    mask_pixels = int(np.sum(binary_mask))
    mask_ratio = mask_pixels / total_pixels

    # ===== 规则1：过小目标过滤 =====
    if mask_pixels < min_positive_pixels:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "reason": "too_small_pixels"
        }

    # ===== 规则2：过小占比过滤 =====
    if mask_ratio < min_mask_ratio:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "reason": "too_small_ratio"
        }

    ys, xs = np.where(binary_mask)
    if len(ys) == 0 or len(xs) == 0:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "reason": "empty_mask"
        }

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)

    # ===== 规则3：贴边过滤 =====
    if y_min < margin or y_max > h - margin:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "reason": "touch_y_border"
        }

    if x_min < margin or x_max > w - margin:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "reason": "touch_x_border"
        }

    # ===== 规则4：bbox 过薄 / 过窄过滤 =====
    if bbox_h < min_bbox_h:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "reason": "bbox_too_short"
        }

    if bbox_w < min_bbox_w:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "reason": "bbox_too_narrow"
        }

    # ===== 规则5：连通域过滤 =====
    cc_areas = get_connected_component_areas(binary_mask)

    if len(cc_areas) == 0:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "reason": "no_connected_component"
        }

    largest_cc_area = cc_areas[0]
    top2_cc_sum = sum(cc_areas[:2]) if len(cc_areas) >= 2 else cc_areas[0]

    if largest_cc_area < min_largest_cc_area:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "largest_cc_area": largest_cc_area,
            "top2_cc_sum": top2_cc_sum,
            "reason": "largest_cc_too_small"
        }

    if top2_cc_sum < min_top2_cc_sum:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "largest_cc_area": largest_cc_area,
            "top2_cc_sum": top2_cc_sum,
            "reason": "top2_cc_sum_too_small"
        }

    # ===== 规则6：薄结构过滤 =====
    fill_ratio = mask_pixels / (bbox_h * bbox_w + 1e-6)

    if fill_ratio < min_fill_ratio:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "largest_cc_area": largest_cc_area,
            "top2_cc_sum": top2_cc_sum,
            "fill_ratio": fill_ratio,
            "reason": "too_thin_structure"
        }

    # ===== 规则7：上下结构过滤（新增核心）=====
    y_center = h // 2
    upper_part = binary_mask[:y_center, :]
    lower_part = binary_mask[y_center:, :]

    upper_pixels = int(np.sum(upper_part))
    lower_pixels = int(np.sum(lower_part))

    if upper_pixels < min_upper_pixels or lower_pixels < min_lower_pixels:
        return False, {
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "bbox_h": bbox_h,
            "bbox_w": bbox_w,
            "largest_cc_area": largest_cc_area,
            "top2_cc_sum": top2_cc_sum,
            "fill_ratio": fill_ratio,
            "upper_pixels": upper_pixels,
            "lower_pixels": lower_pixels,
            "reason": "single_side_structure"
        }

    return True, {
        "mask_pixels": mask_pixels,
        "mask_ratio": mask_ratio,
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "largest_cc_area": largest_cc_area,
        "top2_cc_sum": top2_cc_sum,
        "fill_ratio": fill_ratio,
        "upper_pixels": upper_pixels,
        "lower_pixels": lower_pixels,
        "reason": "ok"
    }


def build_one_split(
    split_name,
    case_list,
    ct_dir,
    mask_dir,
    save_csv,
    min_positive_pixels=500,
    min_mask_ratio=0.002,
    margin=10,
    min_bbox_h=40,
    min_bbox_w=20,
    min_largest_cc_area=200,
    min_top2_cc_sum=500,
    min_fill_ratio=0.15,
    min_upper_pixels=50,
    min_lower_pixels=50
):
    rows = []

    total_slices = 0
    original_positive_slices = 0
    kept_slices = 0

    stats_counter = {
        "too_small_pixels": 0,
        "too_small_ratio": 0,
        "touch_y_border": 0,
        "touch_x_border": 0,
        "bbox_too_short": 0,
        "bbox_too_narrow": 0,
        "largest_cc_too_small": 0,
        "top2_cc_sum_too_small": 0,
        "too_thin_structure": 0,
        "single_side_structure": 0,
        "empty_mask": 0,
        "no_connected_component": 0
    }

    for case_name in case_list:
        ct_path = ct_dir / case_name
        mask_path = mask_dir / case_name

        ct = nib.load(str(ct_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()

        if ct.shape != mask.shape:
            print(f"[跳过] {case_name} 维度不一致: {ct.shape} vs {mask.shape}")
            continue

        if len(ct.shape) != 3:
            print(f"[跳过] {case_name} 不是3D体数据: {ct.shape}")
            continue

        depth = ct.shape[2]
        h, w = ct.shape[0], ct.shape[1]

        for z in range(depth):
            total_slices += 1

            mask_slice = mask[:, :, z]
            if np.sum(mask_slice > 0) > 0:
                original_positive_slices += 1

            valid, info = is_valid_mask_slice(
                mask_slice=mask_slice,
                min_positive_pixels=min_positive_pixels,
                min_mask_ratio=min_mask_ratio,
                margin=margin,
                min_bbox_h=min_bbox_h,
                min_bbox_w=min_bbox_w,
                min_largest_cc_area=min_largest_cc_area,
                min_top2_cc_sum=min_top2_cc_sum,
                min_fill_ratio=min_fill_ratio,
                min_upper_pixels=min_upper_pixels,
                min_lower_pixels=min_lower_pixels
            )

            if not valid:
                reason = info["reason"]
                if reason in stats_counter:
                    stats_counter[reason] += 1
                continue

            kept_slices += 1
            rows.append({
                "split": split_name,
                "case_name": case_name,
                "slice_idx": z,
                "height": h,
                "width": w,
                "mask_pixels": info["mask_pixels"],
                "mask_ratio": info["mask_ratio"],
                "bbox_h": info["bbox_h"],
                "bbox_w": info["bbox_w"],
                "largest_cc_area": info["largest_cc_area"],
                "top2_cc_sum": info["top2_cc_sum"],
                "fill_ratio": info["fill_ratio"],
                "upper_pixels": info["upper_pixels"],
                "lower_pixels": info["lower_pixels"]
            })

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "case_name", "slice_idx",
                "height", "width",
                "mask_pixels", "mask_ratio",
                "bbox_h", "bbox_w",
                "largest_cc_area", "top2_cc_sum",
                "fill_ratio", "upper_pixels", "lower_pixels"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[{split_name}] 完成")
    print(f"病例数: {len(case_list)}")
    print(f"总切片数: {total_slices}")
    print(f"原始正样本切片数: {original_positive_slices}")
    print(f"过滤后保留切片数: {kept_slices}")
    print("过滤统计：")
    for k, v in stats_counter.items():
        print(f"  {k:22s}: {v}")
    print(f"保存到: {save_csv}")


def main():
    cfg = load_config()
    ct_dir = Path(cfg["data"]["ct_dir"])
    mask_dir = Path(cfg["data"]["mask_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    results_dir = Path(cfg["data"]["report_dir"])

    # ===== 推荐参数 =====
    min_positive_pixels = 500
    min_mask_ratio = 0.002
    margin = 10

    min_bbox_h = 40
    min_bbox_w = 20

    min_largest_cc_area = 200
    min_top2_cc_sum = 500

    min_fill_ratio = 0.15

    # ===== 新增：上下结构阈值 =====
    min_upper_pixels = 50
    min_lower_pixels = 50

    for split_name in ["train", "val", "test"]:
        txt_path = splits_dir / f"{split_name}.txt"
        case_list = load_case_list(txt_path)
        save_csv = results_dir / f"{split_name}_slices.csv"

        build_one_split(
            split_name=split_name,
            case_list=case_list,
            ct_dir=ct_dir,
            mask_dir=mask_dir,
            save_csv=save_csv,
            min_positive_pixels=min_positive_pixels,
            min_mask_ratio=min_mask_ratio,
            margin=margin,
            min_bbox_h=min_bbox_h,
            min_bbox_w=min_bbox_w,
            min_largest_cc_area=min_largest_cc_area,
            min_top2_cc_sum=min_top2_cc_sum,
            min_fill_ratio=min_fill_ratio,
            min_upper_pixels=min_upper_pixels,
            min_lower_pixels=min_lower_pixels
        )


if __name__ == "__main__":
    main()