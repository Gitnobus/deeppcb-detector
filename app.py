"""步骤 4：Gradio 可视化页面。

用法（cv 环境）:
    python app.py                 # 自动找最近一次训练的 best.pt
    python app.py --weights path/to/best.pt --conf 0.4

页面功能:
  1) 检测演示：从官方测试集抽样或上传图片 -> 模型预测框叠加（可选叠加 GT 真值）
  2) 模板差分：待检图 / 模板图 / 差分二值图三联视图，展示"成对数据"的意义
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from pcbdet import PROJECT_ROOT, load_config
from pcbdet.convert_yolo import load_split

# 6 类缺陷的固定配色 (BGR)
COLORS = [
    (60, 76, 231),    # open      红
    (231, 76, 60),    # short     橙红
    (60, 180, 75),    # mousebite 绿
    (255, 193, 7),    # spur      黄
    (153, 102, 255),  # copper    紫
    (0, 200, 255),    # pin-hole  琥珀
]
GT_COLOR = (255, 255, 255)


def imread_gray(path) -> np.ndarray:
    """读灰度图，并兼容 ultralytics 给 cv2.imread 打补丁后返回 (H,W,1) 的情况。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图片读取失败: {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return img


def find_latest_weights(cfg) -> Path | None:
    """权重选择顺序：配置 default_weights（随项目发布）> outputs/runs 下最近的 best.pt。"""
    default = getattr(cfg, "default_weights", None)
    if default:
        p = cfg.resolve(default)
        if p.exists():
            return p
    runs = PROJECT_ROOT / "outputs" / "runs"
    candidates = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


class DemoState:
    def __init__(self, weights: Path, cfg):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(str(weights))
        self.weights = weights
        # 官方测试集样本（供下拉选择）
        self.samples = {s["image_id"]: s for s in load_split(cfg.data_root, "test")}
        # 转换产物 pairs.jsonl（若存在），用于展示模板图
        self.pairs = {}
        pairs_file = cfg.dataset_dir / "pairs.jsonl"
        if pairs_file.exists():
            for line in pairs_file.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                self.pairs[d["image_id"]] = d


def draw_boxes(img: np.ndarray, boxes: np.ndarray, class_names: list[str],
               label_prefix: str = "") -> np.ndarray:
    """boxes: [N,6](x1,y1,x2,y2,conf,cls) 或 [N,5](x1,y1,x2,y2,type1_6，GT)。"""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        if len(b) == 6:
            x1, y1, x2, y2, conf, k = b
            color = COLORS[int(k) % len(COLORS)]
            label = f"{label_prefix}{class_names[int(k)]} {conf:.2f}"
        else:  # GT：type 1~6
            x1, y1, x2, y2, t = b
            color = GT_COLOR
            label = f"{label_prefix}GT:{class_names[int(t) - 1]}"
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, color, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        y_text = max(p1[1] - 3, th + 2)
        cv2.rectangle(vis, (p1[0], y_text - th - 2), (p1[0] + tw, y_text + 1), color, -1)
        cv2.putText(vis, label, (p1[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def diff_view(sample: dict) -> np.ndarray:
    """待检图 / 模板图 / 差分二值图 三联图。"""
    test = imread_gray(sample["img_path"])
    temp = imread_gray(sample["temp_path"])
    diff = cv2.absdiff(test, temp)
    _, diff_bin = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    h, w = test.shape
    sep = np.full((h, 4), 128, dtype=np.uint8)
    tri = np.concatenate([test, sep, temp, sep, diff_bin], axis=1)
    return cv2.cvtColor(tri, cv2.COLOR_GRAY2BGR)


def build_app(state: DemoState) -> gr.Blocks:
    cfg = state.cfg
    class_names = cfg.class_names
    ids = sorted(state.samples.keys())

    with gr.Blocks(title="DeepPCB 缺陷检测 Demo") as demo:
        gr.Markdown(
            f"## DeepPCB PCB 缺陷检测 Demo\n"
            f"模型: `{state.weights.name}`（6 类缺陷: {', '.join(class_names)}）"
        )

        with gr.Tab("检测演示"):
            with gr.Row():
                with gr.Column(scale=1):
                    sample_dd = gr.Dropdown(ids, value=ids[0], label="选择官方测试集样本")
                    conf_slider = gr.Slider(0.05, 0.95, value=cfg.app.conf_threshold,
                                            step=0.05, label="置信度阈值")
                    show_gt = gr.Checkbox(value=True, label="叠加 GT 真值框（白色）")
                    btn = gr.Button("开始检测", variant="primary")
                with gr.Column(scale=2):
                    out_img = gr.Image(label="预测结果", interactive=False)
                    out_text = gr.Textbox(label="检测统计", lines=6)

            def detect(image_id: str, conf: float, with_gt: bool):
                s = state.samples[image_id]
                img = imread_gray(s["img_path"])
                if getattr(state, "threeway", False):  # E6 差分模型：现算 3 通道输入
                    from pcbdet.evaluate import make_threeway_input
                    inp = make_threeway_input(s["img_path"], s["temp_path"])
                else:
                    inp = img
                r = state.model.predict(inp, conf=float(conf),
                                        imgsz=cfg.eval.imgsz, verbose=False)[0]
                if r.boxes is None or len(r.boxes) == 0:
                    pred = np.zeros((0, 6))
                else:
                    pred = np.concatenate(
                        [r.boxes.xyxy.cpu().numpy(),
                         r.boxes.conf.cpu().numpy()[:, None],
                         r.boxes.cls.cpu().numpy()[:, None].astype(float)], axis=1)
                vis = draw_boxes(img, pred, class_names)
                if with_gt and len(s["boxes"]):
                    vis = draw_boxes(vis, s["boxes"], class_names, label_prefix="")
                n_by_cls = {name: int((pred[:, 5] == i).sum())
                            for i, name in enumerate(class_names)} if len(pred) else {}
                txt = f"预测框数: {len(pred)}   GT 框数: {len(s['boxes'])}\n"
                txt += "  ".join(f"{k}:{v}" for k, v in n_by_cls.items() if v) or "（无检出）"
                return vis, txt

            btn.click(detect, [sample_dd, conf_slider, show_gt], [out_img, out_text])
            sample_dd.change(detect, [sample_dd, conf_slider, show_gt], [out_img, out_text])
            conf_slider.release(detect, [sample_dd, conf_slider, show_gt], [out_img, out_text])

        with gr.Tab("模板差分"):
            with gr.Row():
                dd2 = gr.Dropdown(ids, value=ids[0], label="选择样本")
                out_diff = gr.Image(label="待检图 | 模板图 | 差分二值图（阈值40）", interactive=False)

            def show_diff(image_id: str):
                s = state.samples[image_id]
                return diff_view(s)

            dd2.change(show_diff, [dd2], [out_diff])

        gr.Markdown(
            "> 数据来源: DeepPCB（Tang et al., arXiv:1902.06197）。"
            "图像为线阵 CCD 扫描二值化后的 640×640 灰度图。"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepPCB 缺陷检测可视化页面")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None,
                        help="缺省用配置 default_weights（发布权重），其次最近一次训练 best.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--diff", action="store_true",
                        help="加载 E6 三通道差分模型（推理时按图对现算 待检/模板/差分 输入）")
    parser.add_argument("--data-root", default=None,
                        help="DeepPCB 数据根目录（含 PCBData/）；也可用环境变量 DEEPPCB_ROOT")
    args = parser.parse_args()

    cfg = load_config(args.config, data_root=args.data_root)
    cfg.check_data_root()
    weights = args.weights or find_latest_weights(cfg)
    if weights is None:
        raise SystemExit("未找到训练权重，请先运行 scripts/train.py，或用 --weights 指定")
    print(f"加载模型: {weights}")
    state = DemoState(Path(weights), cfg)
    state.threeway = args.diff
    demo = build_app(state)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
