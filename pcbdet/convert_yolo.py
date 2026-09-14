"""DeepPCB 原始数据 -> YOLO 目标检测格式转换。

官方清单 trainval.txt / test.txt 使用旧命名（如 group00041/00041/00041000.jpg），
而仓库实际图片已拆分为 `00041000_test.jpg`（待检图）与 `00041000_temp.jpg`（模板图），
本模块负责做命名映射并生成：

    <out>/
    ├── images/{train,val}/xxx_test.jpg   # 直接复制原始待检图（灰度图由 dataloader 转 3 通道）
    ├── labels/{train,val}/xxx_test.txt   # YOLO txt: cls cx cy w h（归一化，cls 从 0 起）
    ├── pcb.yaml                          # ultralytics 数据集配置
    ├── dataset_stats.json                # 样本/框数统计
    └── pairs.jsonl                       # image_id -> (img, temp, gt) 路径，供可视化页面使用
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import Config, load_config

IMG_SIZE = 640  # DeepPCB 图像固定 640x640


def load_split(data_root: Path, split: str) -> list[dict]:
    """解析官方清单，返回样本列表。

    每个样本: dict(image_id, img_path, temp_path, ann_path, boxes=[N,5])
    boxes 为原始像素坐标 [x1, y1, x2, y2, type]（type 1~6）。
    """
    list_file = data_root / "PCBData" / f"{split}.txt"
    if not list_file.exists():
        raise FileNotFoundError(f"清单不存在: {list_file}")

    samples = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        img_ref, ann_ref = line.split()
        # 官方清单是旧命名 xxx.jpg，实际文件为 xxx_test.jpg / xxx_temp.jpg
        img_path = data_root / "PCBData" / img_ref.replace(".jpg", "_test.jpg")
        temp_path = data_root / "PCBData" / img_ref.replace(".jpg", "_temp.jpg")
        ann_path = data_root / "PCBData" / ann_ref
        if not img_path.exists():
            raise FileNotFoundError(f"待检图缺失（清单路径未映射？）: {img_path}")
        if not temp_path.exists():
            raise FileNotFoundError(f"模板图缺失: {temp_path}")

        raw = np.loadtxt(ann_path, dtype=np.float64).reshape(-1, 5)
        boxes = raw[~np.isnan(raw).any(axis=1)] if raw.size else raw.reshape(0, 5)
        samples.append(
            dict(
                image_id=ann_path.stem,  # 如 00041000
                img_path=img_path,
                temp_path=temp_path,
                ann_path=ann_path,
                boxes=boxes,
            )
        )
    return samples


def _write_label(boxes: np.ndarray, dst: Path) -> None:
    lines = []
    for x1, y1, x2, y2, t in boxes:
        cx = (x1 + x2) / 2.0 / IMG_SIZE
        cy = (y1 + y2) / 2.0 / IMG_SIZE
        w = (x2 - x1) / IMG_SIZE
        h = (y2 - y1) / IMG_SIZE
        lines.append(f"{int(t) - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(out_dir: Path, class_names: list[str]) -> Path:
    """生成 ultralytics 数据集配置 pcb.yaml（path 字段随当前目录动态解析，目录移动后重跑即可修复）。"""
    yaml_path = out_dir / "pcb.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path


def build_diff_dataset(cfg: Config, out_dir: Path | None = None,
                       diff_thresh: int = 40) -> dict:
    """E6：构建 3 通道数据集，通道 = (待检图, 模板图, 差分二值图)。

    利用 DeepPCB 的成对信息：absdiff(test, temp) 显式标出"与正常版不同"的区域。
    图像打包成单张 3 通道 JPG，ultralytics 数据管线无需任何改动；标签/划分与官方一致。
    训练时建议关闭 HSV 增广（train.py --no-hsv），避免扰动差分通道语义。

    diff_thresh：差分图二值化阈值，用于滤除 JPEG 压缩噪声（与可视化页面一致取 40）。

    注意：Windows 下 cv2.imread/imwrite 不支持含中文的绝对路径，
    这里统一用 imdecode/imencode + numpy 的文件读写（unicode 安全）。
    """
    import cv2  # 本函数需要 OpenCV

    out_dir = Path(out_dir) if out_dir else cfg.dataset_dir.parent / "pcb_diff"

    def _gray(p: Path) -> np.ndarray:
        buf = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图片读取失败: {p}")
        return img[..., 0] if img.ndim == 3 else img

    def _save_png(p: Path, img: np.ndarray) -> None:
        # PNG 无损：差分通道是二值信号，JPEG 有损压缩会把它涂抹成灰边（实测纯白像素仅剩 0.06%）
        ok, enc = cv2.imencode(".png", img)
        if not ok:
            raise IOError(f"PNG 编码失败: {p}")
        enc.tofile(str(p))

    stats = {"class_names": cfg.class_names, "diff_thresh": diff_thresh, "splits": {}}
    for split in ("trainval", "test"):
        yolo_split = "train" if split == "trainval" else "val"
        img_dir = out_dir / "images" / yolo_split
        lbl_dir = out_dir / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        samples = load_split(cfg.data_root, split)
        n_boxes = 0
        for s in samples:
            test = _gray(s["img_path"])
            temp = _gray(s["temp_path"])
            diff = cv2.absdiff(test, temp)
            _, diff_bin = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
            img3 = cv2.merge([test, temp, diff_bin])  # BGR 语义: B=test, G=temp, R=diff
            name = s["img_path"].stem + "_3ch.png"
            _save_png(img_dir / name, img3)
            _write_label(s["boxes"], lbl_dir / (Path(name).stem + ".txt"))
            n_boxes += len(s["boxes"])

        stats["splits"][yolo_split] = {"images": len(samples), "boxes": int(n_boxes)}
        print(f"[{split} -> {yolo_split}] 3通道图 {len(samples)} 张, 框 {n_boxes} 个")

    yaml_path = write_dataset_yaml(out_dir, cfg.class_names)
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"数据集配置: {yaml_path}")
    return stats


def build_dataset(cfg: Config, out_dir: Path | None = None) -> dict:
    """执行转换，返回统计信息。"""
    out_dir = Path(out_dir) if out_dir else cfg.dataset_dir
    class_names = cfg.class_names

    stats = {"class_names": class_names, "splits": {}}
    pairs_lines = []
    for split in ("trainval", "test"):
        yolo_split = "train" if split == "trainval" else "val"
        img_dir = out_dir / "images" / yolo_split
        lbl_dir = out_dir / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        samples = load_split(cfg.data_root, split)
        box_counter = {name: 0 for name in class_names}
        for s in samples:
            shutil.copy2(s["img_path"], img_dir / s["img_path"].name)
            _write_label(s["boxes"], lbl_dir / (s["img_path"].stem + ".txt"))
            for b in s["boxes"]:
                box_counter[class_names[int(b[4]) - 1]] += 1
            pairs_lines.append(
                json.dumps(
                    {
                        "image_id": s["image_id"],
                        "split": yolo_split,
                        "img": str(s["img_path"]),
                        "temp": str(s["temp_path"]),
                        "gt": str(s["ann_path"]),
                    },
                    ensure_ascii=False,
                )
            )

        stats["splits"][yolo_split] = {
            "images": len(samples),
            "boxes_total": int(sum(box_counter.values())),
            "boxes_per_class": box_counter,
        }
        print(f"[{split} -> {yolo_split}] 图片 {len(samples)} 张, 框 {sum(box_counter.values())} 个")

    # ultralytics 数据集配置
    yaml_path = write_dataset_yaml(out_dir, class_names)

    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pairs.jsonl").write_text("\n".join(pairs_lines) + "\n", encoding="utf-8")
    print(f"数据集配置: {yaml_path}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepPCB -> YOLO 数据转换")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/config.yaml）")
    parser.add_argument("--out", default=None, help="输出目录（默认取配置 dataset_dir）")
    args = parser.parse_args()

    build_dataset(load_config(args.config), args.out and Path(args.out))
