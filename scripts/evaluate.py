"""步骤 3：Python3 官方协议评估（同类 IoU>0.33, mAP + F-score）。

用法（cv 环境）:
    python scripts/evaluate.py --weights outputs/runs/pcb_yolov8n/weights/best.pt
    python scripts/evaluate.py --weights best.pt --conf 0.4 --export-zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcbdet import load_config
from pcbdet import load_config
from pcbdet.evaluate import evaluate_weights


def pick_default_weights(cfg) -> Path | None:
    """权重选择顺序：配置 default_weights（随项目发布）> outputs/runs 下最近的 best.pt。"""
    default = getattr(cfg, "default_weights", None)
    if default:
        p = cfg.resolve(default)
        if p.exists():
            return p
    runs = cfg.output_dir / "runs"
    candidates = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepPCB 官方协议评估（Python3）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None,
                        help="训练权重；缺省用配置 default_weights（发布权重），其次最近一次 best.pt")
    parser.add_argument("--split", default="test", choices=["test", "trainval"])
    parser.add_argument("--iou", type=float, default=None, help="默认 0.33（官方协议）")
    parser.add_argument("--conf", type=float, default=None, help="F-score 报告用阈值，默认 0.25")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="推理输入尺寸，默认取配置 eval.imgsz（须与训练一致）")
    parser.add_argument("--device", default="0", help='GPU id 或 "cpu"')
    parser.add_argument("--export-zip", action="store_true", help="导出官方 res.zip 格式")
    parser.add_argument("--diff", action="store_true",
                        help="评估 E6 三通道差分模型（按图对现算 待检/模板/差分 输入）")
    parser.add_argument("--data-root", default=None,
                        help="DeepPCB 数据根目录（含 PCBData/）；也可用环境变量 DEEPPCB_ROOT")
    args = parser.parse_args()

    cfg = load_config(args.config, data_root=args.data_root)
    cfg.check_data_root()
    weights = args.weights
    if not weights:
        default = pick_default_weights(cfg)
        if default is None:
            raise SystemExit("未找到训练权重，请先训练或用 --weights 指定 best.pt")
        weights = default
        print(f"未指定 --weights，自动使用: {weights}")
    evaluate_weights(
        cfg, weights, split=args.split, iou_thr=args.iou, conf=args.conf,
        imgsz=args.imgsz if args.imgsz is not None else cfg.eval.imgsz,
        device=args.device, export_zip=args.export_zip, threeway=args.diff,
    )


if __name__ == "__main__":
    main()
