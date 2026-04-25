from pathlib import Path
import random
import yaml

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_nii_gz_files(folder: Path):
    return sorted([p for p in folder.glob("*.nii.gz") if p.is_file()])


def main():
    cfg = load_config()
    ct_dir = Path(cfg["data"]["ct_dir"])
    mask_dir = Path(cfg["data"]["mask_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    random.seed(SEED)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not ct_dir.exists():
        raise FileNotFoundError(f"CT目录不存在: {ct_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask目录不存在: {mask_dir}")

    ct_files = list_nii_gz_files(ct_dir)
    mask_files = list_nii_gz_files(mask_dir)

    if len(ct_files) == 0:
        raise RuntimeError(f"CT目录中没有找到 .nii.gz 文件: {ct_dir}")
    if len(mask_files) == 0:
        raise RuntimeError(f"Mask目录中没有找到 .nii.gz 文件: {mask_dir}")

    ct_names = {p.name: p for p in ct_files}
    mask_names = {p.name: p for p in mask_files}

    common_names = sorted(set(ct_names.keys()) & set(mask_names.keys()))
    ct_only = sorted(set(ct_names.keys()) - set(mask_names.keys()))
    mask_only = sorted(set(mask_names.keys()) - set(ct_names.keys()))

    if ct_only:
        print("以下CT没有对应Mask：")
        for x in ct_only[:20]:
            print("  ", x)

    if mask_only:
        print("以下Mask没有对应CT：")
        for x in mask_only[:20]:
            print("  ", x)

    if len(common_names) == 0:
        raise RuntimeError("没有找到任何同名的 CT/Mask 配对病例。")

    random.shuffle(common_names)

    n = len(common_names)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_names = common_names[:n_train]
    val_names = common_names[n_train:n_train + n_val]
    test_names = common_names[n_train + n_val:]

    save_list(splits_dir / "train.txt", train_names)
    save_list(splits_dir / "val.txt", val_names)
    save_list(splits_dir / "test.txt", test_names)

    print("\n划分完成")
    print(f"可配对病例总数: {n}")
    print(f"train: {len(train_names)}")
    print(f"val:   {len(val_names)}")
    print(f"test:  {len(test_names)}")
    print(f"划分文件已保存到: {splits_dir}")


def save_list(path: Path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item + "\n")


if __name__ == "__main__":
    main()