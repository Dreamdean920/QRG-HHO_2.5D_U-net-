from pathlib import Path
import csv
import yaml
import nibabel as nib
import numpy as np


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_names(splits_dir: Path):
    all_names = set()
    for split_name in ["train.txt", "val.txt", "test.txt"]:
        split_path = splits_dir / split_name
        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        all_names.add(name)
    return sorted(all_names)


def check_one(ct_path: Path, mask_path: Path):
    row = {
        "case_name": ct_path.name,
        "ct_exists": ct_path.exists(),
        "mask_exists": mask_path.exists(),
        "ct_shape": "",
        "mask_shape": "",
        "shape_match": False,
        "ct_dtype": "",
        "mask_dtype": "",
        "ct_min": "",
        "ct_max": "",
        "mask_unique_values_sample": "",
        "empty_mask": "",
        "remark": ""
    }

    if not ct_path.exists():
        row["remark"] = "CT不存在"
        return row

    if not mask_path.exists():
        row["remark"] = "Mask不存在"
        return row

    try:
        ct_obj = nib.load(str(ct_path))
        mask_obj = nib.load(str(mask_path))

        ct = ct_obj.get_fdata()
        mask = mask_obj.get_fdata()

        row["ct_shape"] = str(ct.shape)
        row["mask_shape"] = str(mask.shape)
        row["shape_match"] = (ct.shape == mask.shape)

        row["ct_dtype"] = str(ct.dtype)
        row["mask_dtype"] = str(mask.dtype)

        row["ct_min"] = float(np.min(ct))
        row["ct_max"] = float(np.max(ct))

        uniq = np.unique(mask)
        if len(uniq) > 20:
            row["mask_unique_values_sample"] = ",".join(map(str, uniq[:20].tolist())) + "..."
        else:
            row["mask_unique_values_sample"] = ",".join(map(str, uniq.tolist()))

        row["empty_mask"] = bool(np.sum(mask > 0) == 0)

        if not row["shape_match"]:
            row["remark"] = "CT和Mask维度不一致"
        elif row["empty_mask"]:
            row["remark"] = "Mask为空"
        else:
            row["remark"] = "OK"

    except Exception as e:
        row["remark"] = f"读取失败: {e}"

    return row


def main():
    cfg = load_config()

    ct_dir = Path(cfg["data"]["ct_dir"])
    mask_dir = Path(cfg["data"]["mask_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    report_dir = Path(cfg["data"]["report_dir"])

    report_dir.mkdir(parents=True, exist_ok=True)
    result_csv = report_dir / "data_check_report_nii.csv"

    case_names = load_split_names(splits_dir)
    if len(case_names) == 0:
        raise RuntimeError("没有找到划分文件内容，请先运行 make_splits_nii.py")

    rows = []
    ok_count = 0

    for case_name in case_names:
        ct_path = ct_dir / case_name
        mask_path = mask_dir / case_name
        row = check_one(ct_path, mask_path)
        rows.append(row)
        if row["remark"] == "OK":
            ok_count += 1

    with open(result_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("数据检查完成")
    print(f"总病例数: {len(rows)}")
    print(f"正常病例数: {ok_count}")
    print(f"报告已保存到: {result_csv}")

    bad_rows = [r for r in rows if r["remark"] != "OK"]
    if bad_rows:
        print("\n存在问题的病例：")
        for r in bad_rows[:20]:
            print(f"{r['case_name']} -> {r['remark']}")
    else:
        print("所有病例检查通过。")


if __name__ == "__main__":
    main()