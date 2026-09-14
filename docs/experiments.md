# TODO · DeepPCB_demo 后续优化实验

> 记录待尝试的改进项。基线 `default_conf`（yolov8n/640）结果：
> mAP@0.33 = 97.42%，F-score 94.94% @ conf 0.70（详见 DeepPCB_detect/训练结果分析_default_conf.md）。
> 每完成一项，把结果补到对应条目下。

## 已可直接执行的实验

- [ ] **E1 · 换 yolov8s 模型**
  ```powershell
  python scripts/train.py --model yolov8s.pt --name yolov8s_conf
  python scripts/evaluate.py --weights outputs/runs/yolov8s_conf/weights/best.pt --conf 0.7
  ```
  首次运行自动下载 yolov8s.pt（约 22MB）；显存不足时加 `--batch 16`。
  预计训练时长约为 yolov8n 的 3 倍（40~60 分钟）。
  - 结果（2024-09 实测，best epoch 48）：**未超过 yolov8n 基线**——
    官方协议 mAP@0.33 = 96.83%（基线 97.42%，-0.59）；
    F-score 最优 94.78% @ conf 0.56（基线 94.94% @ 0.70）；
    ultralytics mAP50 0.9767 / mAP50-95 0.7598（mAP50 略升，mAP50-95 基本持平）。
    问题：pin-hole AP 从 96.61 跌到 92.42（新增 pin-hole→mousebite 0.20 混淆），
    mousebite→open 混淆从 0.27 升到 0.33；best epoch 早至 48、之后持续退化，
    呈现过拟合倾向（训练集仅 1,000 张，小模型反而泛化更好）。
    训练耗时 1,565s（基线 976s，+60%）。**结论：容量不是瓶颈，维持 yolov8n，
    优先做 E2/E3/E6。**

- [ ] **E2 · 关闭/调整 mosaic 增广**（观察后期 mAP50-95 震荡是否消失）
  ```powershell
  python scripts/train.py --mosaic 0 --name n_nomosaic          # 全程关闭
  python scripts/train.py --close-mosaic 30 --name n_cm30       # 或：前 70 epoch 开启
  ```
  - 结果（全程关闭 mosaic=0，best epoch 77，耗时 999s）：
    官方协议 mAP@0.33 = 97.08%（基线 97.42%，-0.34）；
    **F-score 最优 95.39% @ conf 0.69（基线 94.94%，+0.45，召回 0.938→0.952）**；
    检测框总数 22,852→**9,703**（低置信度冗余框大幅减少，置信度校准明显改善）；
    mAP50-95 best 0.7703（基线 0.7630）、末轮 0.7527（基线 0.7239）、尾段 std 0.037（基线 0.044）
    ——定位精度与后期稳定性均更好；val_box_loss 末轮 0.819 vs 0.920。
    代价：pin-hole AP 96.61→94.37。
    **结论：关 mosaic 换来更好的 F-score/校准/稳定性，轻微牺牲宽松口径 mAP。待试方案 B（close_mosaic=30）。**
  - 结果（方案 B：close_mosaic=30，best epoch 88，耗时 1025s）：**当前最优综合方案**——
    官方协议 mAP@0.33 = 97.41%（恢复到基线水平 97.42%）；
    **F-score 最优 95.51% @ conf 0.70（全部实验最高）**；
    pin-hole AP 97.08%（全部实验最高，方案 A 为 94.37）；
    检测框总数 10,304（与方案 A 同样干净）；short 93.68 略低于基线 94.05。
    注意：尾段 mAP50-95 仍有震荡（std 0.049，与基线相当），说明震荡主因可能不是
    mosaic 切换而是 lr 调度尾段（E7 用 cos_lr 验证）。
    **消融小结：close_mosaic=30 兼得基线的 mAP 与关 mosaic 的 F-score，定为当前默认推荐配置。**

- [x] **E3 · 提高输入分辨率**（cm30 基础上 `--imgsz 960 --batch 16 --close-mosaic 30`）
  - 结果（best epoch 74，耗时 2,023s，约为 cm30 的 2 倍）：**mAP 新高**——
    官方协议 mAP@0.33 = **97.78%**（cm30 97.41 / 基线 97.42，+0.37）；
    **short AP 93.68 → 95.93（+2.25），长期最弱类大幅改善**；
    open 98.87 / spur 97.55 / copper 99.03 同步微升；mousebite 98.59 / pin-hole 96.73 微降；
    F-score 最优 95.20% @ conf 0.66（略低于 cm30 的 95.51），
    但默认阈值 0.25 下 F-score 93.92%（cm30 92.99）——置信度校准更好、对阈值不敏感；
    检测框总数 8,993（最少）；训练尾段 mAP50-95 std **0.018**（最稳定）。
    代价：训练时长翻倍（34 分钟）；推理计算量随像素数增至 2.25 倍
    （约 8.1→18.2 GFLOPs）；部署/评估必须用 imgsz=960。
    **结论：mAP 最优方案；short 类短板基本补齐。**
    （注意：部署 imgsz 需与训练一致，`configs/config.yaml` 的 `eval.imgsz` 同步改 960）

- [ ] **E4 · 最优部署阈值固化**：确定最终模型后，把 conf=0.70 写入 `configs/config.yaml`
  的 `eval.conf_threshold` 与 `app.conf_threshold`。
  - 结果：_待填_

## 需要改代码的实验

- [ ] **E5 · P2 小目标检测头**
  - 背景：YOLOv8 默认在 stride 8/16/32（P3/P4/P5）三个尺度上检测；P2 头增加 stride 4 的
    高分辨率检测层，专门捕捉 <16px 的微小目标（本数据集缺陷框普遍 20~60px）。
  - 做法（ ultralytics 官方提供 yolov8-p2 配置）：
    ```powershell
    yolo detect train model=yolov8n-p2.yaml data=<项目>/datasets/pcb/pcb.yaml imgsz=640 epochs=100 batch=32 project=<项目>/outputs/runs name=n_p2
    ```
    （从 yaml 从头训练；迁移预训练权重需用 python API
    `YOLO("yolov8n-p2.yaml").load("yolov8n.pt").train(...)`，必要时给 train.py 加 `--load` 参数）
  - 结果：_待填_

- [ ] **E6 · 模板差分作为额外输入通道**（差异化亮点，呼应 GPP 论文思路）
  - **已实现（含评估/可视化支持）**：输入为 3 通道 PNG（B=待检图, G=模板图, R=差分二值图 `absdiff>40`），
    数据管线零改动，标签/划分与官方一致。
    - 构建：`python scripts/prepare_data.py --diff` → `datasets/pcb_diff/`（1,500 张 PNG，约 118MB；
      用 PNG 而非 JPEG——实测 JPEG 有损压缩会把二值差分信号涂抹成灰边，纯白像素仅剩 0.06%）
    - 陷阱备忘：Windows 下 `cv2.imwrite/imread` 不支持中文绝对路径（静默失败），
      代码里已改用 `imencode+tofile / fromfile+imdecode`
    - 评估：`scripts/evaluate.py --diff`（按图对现算 3 通道输入，无损）；
      可视化：`python app.py --weights .../best.pt --diff`
  - 训练（--no-hsv 必加，保护差分通道语义；imgsz/close_mosaic 走 config 默认 960/30）：
    ```powershell
    python scripts/train.py --data datasets/pcb_diff/pcb.yaml --no-hsv --name n_diff
    python scripts/evaluate.py --weights outputs/runs/n_diff/weights/best.pt --diff
    ```
  - 结果（best epoch 27，耗时 1,959s）：**全部实验最优，mAP 超过论文 GPP 模型**——
    官方协议 mAP@0.33 = **99.25%**（对比 n_imgsz960 97.78，+1.47；论文 GPP 98.6）；
    **F-score 最优 98.04% @ conf 0.47**（论文 GPP 98.2，基本持平）；
    per-class AP：**pin-hole 100.00%**、copper 99.36、short 99.18（+3.25）、mousebite 99.18、
    spur 98.80、open 98.96——六类全部 ≥98.8，无短板类。
    注意事项：
    ① 过拟合早现：best 在 ep27，之后 mAP50-95 一路退化（末轮 0.577，val_box 1.533），
    后续训练 50 epochs + patience 20 即可；
    ② COCO 口径 mAP50-95（0.754）低于 n_imgsz960（0.792）：高 IoU 下框贴合度略差，
    但官方协议（IoU>0.33）不受影响；
    ③ 低置信度冗余检测仍多（25,151 @ conf 0.001），部署阈值建议 0.47~0.70；
    ④ ultralytics 混淆矩阵（固定 conf 0.25）看似混乱，是低阈值工作点噪声，与 AP 排序无关。
    **结论：模板差分通道是决定性改进；demo 主力模型定为 n_diff。**
- [ ] **E7 · 训练稳定性与早停验证**
  - 背景：n_diff 的 best（ep27）是 mAP50-95 的单轮尖峰（0.754，前后轮均 ~0.42），
    疑似噪声快照；且训练后期持续退化（末轮 val_box 1.53）。
    注意：**不要用 epochs=50 + close_mosaic=30 复现**——mosaic 会在第 21 轮关闭，
    与 n_diff 最优点（mosaic 开启阶段）矛盾；正确做法是 epochs=100 + patience=20 早停（约 47 轮收敛）。
  - 结果（n_diff_v2，2024-09 实测）：**早停精确生效且结果完全复现**——
    第 47 轮触发早停（27+20），耗时 894s（n_diff 1,959s 的 46%）；
    best 仍在 ep27，官方协议 mAP@0.33 = 99.25%、F-score 98.04% @ 0.47，
    per-class AP、检测框总数（25,151）与 n_diff 逐项一致。
    原因：seed=42 + deterministic=True 下训练完全确定性，v2 即同一条轨迹的截断版，
    best.pt 与 n_diff 等价 → **早停配置可放心作为默认（时间减半，零精度损失）**。
    遗留→已验证（跨种子实验 s1/s2）：
    - seed 1：best ep19，39 轮早停，mAP@0.33 = **99.32%**、F-score 98.26%——与 seed 42 高度一致，配方稳定；
    - seed 2：best 被锚定在 ep3 的 fitness 单轮尖峰（mAP50-95 0.673，此后再未超越），
      第 23 轮被早停误杀，此时 mAP50 仍在上升（ep20 已达 0.984 > ep3 的 0.957），mAP 仅 96.09%
      ——**属早停判据脆弱（fitness 被 mAP50-95 噪声主导）导致的欠训练，非方法本身不稳**；
    - 修正：config 默认 patience 20 → 50（异常尖峰也有机会恢复，正常收敛时仅多跑十几轮）；
    - 可选 E9 → 已验证（n_diff_s2_patience50）：patience=50 下 seed 2 训练到 86 轮、
      best ep36，mAP@0.33 = **99.36%**、F-score 98.36% @ 0.55——**四个实验最高**，
      完全证实 seed 2 此前的 96.09% 是早停误杀所致；
      三个种子最终水平 99.25~99.36%（均值 99.31%，std 0.05），方法对随机种子鲁棒。

- [ ] **E8 · short 类定向优化**：从 `outputs/runs/<name>/` 的验证预测中导出 short 类
  漏检/误检样本清单，做定向过采样或缺陷粘贴增强（spur↔short 边界样本需在文档中说明）。
  - 结果：_待填_
