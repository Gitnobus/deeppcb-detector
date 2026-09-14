# 训练结果分析 · default_conf（YOLOv8n 基线）

> 运行目录：`DeepPCB_detect/DeepPCB_demo/outputs/runs/default_conf`
> 配置：yolov8n.pt 预训练 / imgsz 640 / batch 32 / 100 epochs / AdamW(auto) / RTX 3060，总耗时约 16 分钟。
> 测试集：官方 test 500 张 / 3,140 GT 框（评估协议：同类 IoU>0.33，与原论文对齐）。

## 1. 结果总览

| 指标 | 本次（YOLOv8n） | 论文参考（DeepPCB 官方） |
|---|---|---|
| mAP@0.33（官方协议） | **97.42%** | 图像处理 89.3 / SSD 95.9 / Faster R-CNN 97.6 / **GPP 98.6** |
| F-score（默认 conf=0.25） | 92.60% | — |
| F-score（最优 conf=0.70） | **94.94%**（P=0.961, R=0.938） | GPP 模型 98.2 |
| ultralytics mAP50 / mAP50-95 | 0.972 / 0.724（best epoch 82） | — |

per-class AP@0.33（测试集）：

| 类别 | AP | 备注 |
|---|---|---|
| open | 98.89% | 最好 |
| mousebite | 98.88% | |
| copper | 98.76% | |
| spur | 97.32% | 与 short 互相混淆 |
| pin-hole | 96.61% | |
| **short** | **94.05%** | 最弱，明显拖后腿 |

**定位**：基线合格——超过 SSD，略低于 Faster R-CNN，距论文 GPP 模型（98.6）差约 1.2 个百分点。
继续优化到 98%+ 是现实可行的（YOLOv8 系列在该数据集上普遍能到 98+）。

## 2. 存在的问题（按影响排序）

### 2.1 short 类明显偏弱，且类别混淆集中在"形态相近"的缺陷间

归一化混淆矩阵（行为预测、列为真值）：

- `mousebite → open` 0.27、`open → mousebite` 0.15：两种都是"导线边缘异常"，二值图上形态接近；
- `mousebite → copper` 0.23、`mousebite → pin-hole` 0.23：mousebite 大量被误判为其他类，是分类混乱的中心；
- `short → spur` 0.07、`spur → short` 0.12：short 是"多余连接"、spur 是"边缘凸起"，凸起长到搭上另一根线就是 short，形态连续过渡导致边界样本天然易混。

### 2.2 低置信度冗余检测多，置信度校准不佳

- conf=0.001 全量统计：22,852 个检测 vs 3,140 GT（**7.3 倍**）；
- 部署阈值 0.25 时 F-score 只有 92.6%，把阈值提到 **0.70** 才到最优 94.9%——说明模型给大量背景/错误框打出了中等偏高的置信度，默认阈值下假阳性严重（val_batch0_pred.jpg 中可见同一位置堆叠多个重复框）。

### 2.3 定位精度不足，mAP50-95 后期剧烈震荡

- mAP50-95 = 0.724，与 mAP50 = 0.972 差距大：缺陷是细长小目标，框边缘差几个像素在高 IoU 阈值下 AP 就大幅衰减；
- epoch 81–100 的 mAP50-95 在 **0.575–0.763** 之间跳动（std=0.044），best epoch 停在 82，后期没能超越；
- 后期 val/box_loss 仍周期性冲高（ep95 1.09 vs ep80 0.88），与 mosaic 增广在 `close_mosaic=10` 时突然关闭造成的分布切换、以及学习率尾段调度有关。

### 2.4 训练配置为通用默认值，未针对二值 PCB 图适配

- 默认 HSV/色彩抖动对 0/255 二值图基本无效（无害但无益）；
- mosaic 拼接对"整图内相对位置有语义"的 PCB 裁剪块收益存疑；
- yolov8n 只有 3.0M 参数、8.1 GFLOPs，对这个已对齐、已二值化的"简单"任务可能容量不足。

## 3. 改进建议（按性价比排序）

1. **立刻可做（零成本）**：部署/评估阈值改用 0.70（`evaluate.py --conf 0.7`、`app.py --conf 0.7`），F-score 直接 +2.3 个百分点；也可在 app 页面按类调阈值。
2. **换更大模型**：`--model yolov8s.pt`（11.2M 参数）甚至 yolov8m，12GB 显存足够，预计缩短与 98.6 的差距；这是最省力的高收益项。
3. **针对性调增广**（改 `scripts/train.py` 里 `model.train(...)` 的参数）：
   - `mosaic=0.0`（或提前 `close_mosaic=30`）观察后期震荡是否消失；
   - `hsv_h=0, hsv_s=0, hsv_v=0` 关掉无效的色彩抖动；
   - 保留 fliplr/flipud（PCB 翻转是合理增强），可试 `degrees=5` 小角度旋转。
4. **提升小目标定位**：`imgsz` 提到 800/960（缺陷框普遍只有 20~60 px）；或使用带 P2 检测头的 yolov8-p2 变体，专门针对小目标。
5. **利用模板差分（差异化亮点，呼应 GPP 论文思路）**：把 `absdiff(test, temp)` 二值差分图作为第三输入通道（或直接以差分图训练），显式利用 DeepPCB 的成对信息，预期对边缘类缺陷（mousebite/spur/open）帮助最大。
6. **针对 short 类**：分析 short 漏检/误检样本后，可对 short 做定向过采样或缺陷粘贴增强；类别间边界样本建议在 README 中说明（spur↔short 在形态上本就连续）。
7. **训练细节**：`cos_lr=True` + epochs 150 观察尾部稳定性；`patience=30` 早停避免无效训练。

## 4. 报告口径提醒

- 论文 GPP 的 98.6 mAP 用的是 **IoU>0.33 的宽松口径**，与 COCO 常规 mAP50:95 不可比；
- 本报告的 mAP@0.33（97.42）与论文同口径可直接对比；ultralytics 自带的 mAP50/mAP50-95 是 COCO 口径，写报告时分开列出。
