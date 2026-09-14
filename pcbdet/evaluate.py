"""Python3 版 DeepPCB 官方评估协议实现。

对齐论文《Online PCB Defect Detector On A New PCB Defect Dataset》§2.3：
- 命中判定：检测框与同类别 GT 框 IoU > 0.33（阈值可配置）
- 指标：mAP（各类 AP 的均值）+ F-score（F = 2PR / (P+R)，并搜索最优置信度阈值）
- 匹配方式与 VOC 一致：全部检测按置信度降序贪心匹配，每个 GT 至多被匹配一次

另支持把检测结果导出为官方 res.zip 格式（x1,y1,x2,y2,confidence,type），
type 为 open/short/mousebite/spur/copper/pin-hole，供官方脚本交叉校验。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .convert_yolo import load_split

OFFICIAL_CLASS_NAMES = {  # 官方评估协议使用的类别字符串
    "open": "open",
    "short": "short",
    "mousebite": "mousebite",
    "spur": "spur",
    "copper": "copper",
    "pin-hole": "pin-hole",
}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [N,4] (x1,y1,x2,y2), b: [M,4] -> [N,M] IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def load_gt_boxes(cfg: Config, split: str = "test") -> dict[str, np.ndarray]:
    """返回 {image_id: [N,5](x1,y1,x2,y2,cls0)}，cls0 = 官方 type - 1。"""
    gt: dict[str, np.ndarray] = {}
    for s in load_split(cfg.data_root, split):
        boxes = s["boxes"]
        arr = np.zeros((len(boxes), 5), dtype=np.float64)
        if len(boxes):
            arr[:, :4] = boxes[:, :4]
            arr[:, 4] = boxes[:, 4] - 1  # 转 0 基
        gt[s["image_id"]] = arr
    return gt


def imread_gray(path) -> np.ndarray:
    """unicode 安全 + 兼容 ultralytics 给 cv2.imread 打补丁后返回 (H,W,1) 的情况。"""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图片读取失败: {path}")
    return img[..., 0] if img.ndim == 3 else img


def make_threeway_input(img_path, temp_path, diff_thresh: int = 40) -> np.ndarray:
    """E6 差分模型输入：3 通道 (待检图, 模板图, 差分二值图)，与训练数据构建逻辑一致。"""
    import cv2

    test = imread_gray(img_path)
    temp = imread_gray(temp_path)
    _, diff_bin = cv2.threshold(cv2.absdiff(test, temp), diff_thresh, 255, cv2.THRESH_BINARY)
    return cv2.merge([test, temp, diff_bin])


def predict_dataset(model, image_paths: list[tuple[str, Path]], conf: float = 0.001,
                    imgsz: int = 640, device=0,
                    threeway_samples: list[dict] | None = None) -> dict[str, np.ndarray]:
    """批量推理。返回 {image_id: [N,6](x1,y1,x2,y2,conf,cls0)}。

    threeway_samples 不为 None 时（E6 差分模型），输入为按图对现算的 3 通道数组。
    """
    if threeway_samples is None:
        inputs = [str(p) for _, p in image_paths]
    else:
        inputs = [make_threeway_input(s["img_path"], s["temp_path"])
                  for s in threeway_samples]
    results: dict[str, np.ndarray] = {}
    stream = model.predict(inputs, conf=conf, imgsz=imgsz, device=device,
                           verbose=False, stream=True)
    for (image_id, _), r in zip(image_paths, stream):
        if r.boxes is None or len(r.boxes) == 0:
            results[image_id] = np.zeros((0, 6))
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.conf.cpu().numpy()
        k = r.boxes.cls.cpu().numpy().astype(int)
        results[image_id] = np.concatenate([xyxy, c[:, None], k[:, None]], axis=1)
    return results


def _match(flat: list[tuple], gt: dict[str, np.ndarray], iou_thr: float, n_cls: int):
    """flat: [(conf, cls, image_id, box)]（须已按置信度降序）。

    返回:
      cls_seq: {cls: [(conf, tp)]}  该类检测按置信度降序
      all_seq: [(conf, tp)]         全部检测按置信度降序（算总体 F-score 用）
    """
    used_gt: set = set()
    cls_seq: dict[int, list[tuple[float, int]]] = {c: [] for c in range(n_cls)}
    all_seq: list[tuple[float, int]] = []
    for conf, k, image_id, box in flat:
        g = gt.get(image_id, np.zeros((0, 5)))
        gk = g[g[:, 4] == k] if len(g) else np.zeros((0, 5))
        is_tp = False
        if len(gk):
            ious = iou_matrix(np.asarray(box)[None, :], gk[:, :4])[0]
            j = int(np.argmax(ious))
            if ious[j] > iou_thr and (image_id, k, j) not in used_gt:
                used_gt.add((image_id, k, j))
                is_tp = True
        cls_seq[k].append((conf, is_tp))
        all_seq.append((conf, is_tp))
    return cls_seq, all_seq


def _ap_from_tps(tps: np.ndarray, n_gt: int) -> float:
    """tps（该类检测的 tp 标记，已按置信度降序）-> AP（全点插值 PR 曲线下面积）。"""
    if n_gt == 0 or len(tps) == 0:
        return 0.0
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1 - tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):  # 单调化
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _fscore_at(all_seq: list[tuple[float, int]], thr: float, n_gt_total: int):
    tps = np.array([tp for conf, tp in all_seq if conf >= thr], dtype=np.float64)
    tp = float(tps.sum()) if len(tps) else 0.0
    fp = float(len(tps)) - tp
    p = tp / max(tp + fp, 1e-12)
    r = tp / max(n_gt_total, 1e-12)
    f = 2 * p * r / max(p + r, 1e-12)
    return p, r, f


def compute_metrics(preds: dict[str, np.ndarray], gt: dict[str, np.ndarray],
                    class_names: list[str], iou_thr: float = 0.33,
                    conf_for_fscore: float = 0.25) -> dict:
    """计算 mAP 与 F-score（含最优阈值搜索）。"""
    n_cls = len(class_names)
    n_gt_per_cls = {c: 0 for c in range(n_cls)}
    for arr in gt.values():
        for k in arr[:, 4].astype(int):
            n_gt_per_cls[k] += 1

    flat = []
    for image_id, arr in preds.items():
        for x1, y1, x2, y2, conf, k in arr:
            flat.append((float(conf), int(k), image_id, [x1, y1, x2, y2]))
    flat.sort(key=lambda t: -t[0])

    cls_seq, all_seq = _match(flat, gt, iou_thr, n_cls)

    per_class = {}
    aps = []
    for c, name in enumerate(class_names):
        tps = np.array([tp for _, tp in cls_seq[c]], dtype=np.float64)
        ap = _ap_from_tps(tps, n_gt_per_cls[c])
        aps.append(ap)
        tp_cum = float(tps.sum()) if len(tps) else 0.0
        per_class[name] = dict(
            ap=ap,
            n_gt=n_gt_per_cls[c],
            n_det=len(tps),
            precision=tp_cum / max(len(tps), 1) if len(tps) else 0.0,
            recall=tp_cum / n_gt_per_cls[c] if n_gt_per_cls[c] else 0.0,
        )

    n_gt_total = int(sum(n_gt_per_cls.values()))
    confs = sorted({round(c, 6) for c, _ in all_seq}, reverse=True)
    best = (0.0, 0.0, 0.0, 0.0)  # f, p, r, thr
    for thr in [1e9] + confs:  # 1e9 = 空检测集
        p, r, f = _fscore_at(all_seq, thr, n_gt_total)
        if f > best[0]:
            best = (f, p, r, thr if thr < 1e9 else float("inf"))
    p_def, r_def, f_def = _fscore_at(all_seq, conf_for_fscore, n_gt_total)

    return dict(
        iou_threshold=iou_thr,
        mAP=float(np.mean(aps)) * 100,
        per_class=per_class,
        fscore=dict(
            default_thr=conf_for_fscore, precision=p_def, recall=r_def, fscore=f_def * 100,
            best_thr=best[3], best_precision=best[1], best_recall=best[2], best_fscore=best[0] * 100,
        ),
        n_total_gt=n_gt_total,
        n_total_det=len(flat),
    )


def format_report(m: dict) -> str:
    lines = []
    lines.append(f"IoU 阈值: {m['iou_threshold']}（官方协议 0.33）   GT 框数: {m['n_total_gt']}   检测框数: {m['n_total_det']}")
    lines.append("-" * 76)
    lines.append(f"{'class':<12}{'AP':>8}{'GT':>7}{'DET':>7}{'P':>8}{'R':>8}")
    for name, s in m["per_class"].items():
        lines.append(
            f"{name:<12}{s['ap'] * 100:>7.2f}%{s['n_gt']:>7}{s['n_det']:>7}{s['precision']:>8.4f}{s['recall']:>8.4f}"
        )
    lines.append("-" * 76)
    lines.append(f"mAP@{m['iou_threshold']}: {m['mAP']:.2f}%")
    f = m["fscore"]
    lines.append(
        f"F-score @ conf={f['default_thr']}: {f['fscore']:.2f}%  (P={f['precision']:.4f}, R={f['recall']:.4f})"
    )
    best_thr = f["best_thr"]
    thr_str = "1.0(空集)" if best_thr == float("inf") else f"{best_thr}"
    lines.append(
        f"F-score 最优 @ conf={thr_str}: {f['best_fscore']:.2f}%  (P={f['best_precision']:.4f}, R={f['best_recall']:.4f})"
    )
    return "\n".join(lines)


def export_res_zip(preds: dict[str, np.ndarray], class_names: list[str],
                   out_zip: Path) -> None:
    """导出官方评估脚本格式的 res.zip（每图一个 txt）。"""
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_id, arr in preds.items():
            lines = [
                f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{conf:.4f},{OFFICIAL_CLASS_NAMES[class_names[int(k)]]}"
                for x1, y1, x2, y2, conf, k in arr
            ]
            zf.writestr(f"{image_id}.txt", "\n".join(lines) + ("\n" if lines else ""))
    print(f"已导出官方格式结果: {out_zip}")


def evaluate_weights(cfg: Config, weights: str | Path, split: str = "test",
                     iou_thr: float | None = None, conf: float | None = None,
                     imgsz: int = 640, device=0, export_zip: bool = False,
                     save_json: bool = True, threeway: bool = False) -> dict:
    """完整评估流程：加载模型 -> 推理 -> 计算指标 -> 报告/导出。

    threeway=True 用于 E6 差分模型：按图对现算 3 通道 (待检/模板/差分) 输入。
    """
    from ultralytics import YOLO  # 延迟导入，纯转换场景无需加载 torch

    iou_thr = cfg.eval.iou_threshold if iou_thr is None else iou_thr
    conf_default = cfg.eval.conf_threshold if conf is None else conf

    model = YOLO(str(weights))
    samples = load_split(cfg.data_root, split)
    image_paths = [(s["image_id"], s["img_path"]) for s in samples]
    gt = load_gt_boxes(cfg, split)
    preds = predict_dataset(model, image_paths, conf=0.001, imgsz=imgsz, device=device,
                            threeway_samples=samples if threeway else None)

    metrics = compute_metrics(preds, gt, cfg.class_names,
                              iou_thr=iou_thr, conf_for_fscore=conf_default)
    print(format_report(metrics))

    if export_zip:
        export_res_zip(preds, cfg.class_names, cfg.output_dir / "res.zip")
    if save_json:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.output_dir / f"metrics_{split}.json"
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"指标已保存: {out}")
    return metrics
