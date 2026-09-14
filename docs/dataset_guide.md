# DeepPCB 项目详解（含二次开发指南）

> 本文档基于官方 README 与对本地数据（`F:\data\PCB\DeepPCB-master`）的逐项实测整理而成，
> 在官方 README 基础上补充了：目录结构、数据统计实测、格式细节、官方实现的坑、
> 以及面向 Python 二次开发（开源 demo）的具体路线。
> 所有"实测"字样的数字均为对本地数据集运行脚本统计得到，非官方口径。

- 论文：*On-line PCB Defect Detector On A New PCB Defect Dataset*（数据集随论文发布，论文方法代码未开源）
- License：MIT（官方 README 另注明"仅限研究用途"，做开源 demo 时建议保留原文引用与出处）
- 原始仓库：https://github.com/tangsanli5201/DeepPCB

---

## 1. 项目是什么

DeepPCB 是一个 **PCB（印刷电路板）表面缺陷检测数据集**，核心内容：

| 项目 | 内容 |
|---|---|
| 图像对数量 | **1,500 对**（每对 = 一张无缺陷模板图 + 一张对齐的待检图） |
| 图像尺寸 | 640 × 640，**灰度二值图**（单通道，像素值约 0/255） |
| 成像来源 | 线阵 CCD 扫描，分辨率约 **48 像素 / 毫米**；原始整板约 16k×16k，经模板匹配裁剪对齐成 640×640 子图，并做了阈值二值化以消除光照干扰 |
| 缺陷类别 | 6 类：open（开路）、short（短路）、mousebite（鼠咬）、spur（毛刺）、copper（铜渣/杂铜）、pin-hole（针孔） |
| 标注形式 | 轴对齐矩形框 + 类别 ID（只标待检图，模板图无缺陷不标） |
| 官方划分 | 训练/验证集 1,000 张（`trainval.txt`），测试集 500 张（`test.txt`） |
| 官方成绩 | 其自研模型 98.6% mAP / 98.2% F-score @ 62 FPS（代码未公开） |

典型用法（也是官方方法与大多数后续工作的思路）：
**模板图与待检图对齐后做差分/拼接，再用目标检测网络定位并分类缺陷**，而不是在原始整板上直接检测。

## 2. 目录结构（本地实测）

```
DeepPCB-master/
├── README.md                    # 官方英文说明（本文件是其扩展）
├── LICENSE                      # MIT
├── PCBData/                     # ★ 数据集主体
│   ├── trainval.txt             # 训练/验证集清单（1000 行）
│   ├── test.txt                 # 测试集清单（500 行）
│   ├── group00041/              # 11 个 group 目录（group00041 ~ group92000）
│   │   ├── 00041/
│   │   │   ├── 00041000_temp.jpg   # 模板图（无缺陷）
│   │   │   ├── 00041000_test.jpg   # 待检图（含缺陷）
│   │   │   └── ...
│   │   └── 00041_not/           # ★ "not" = annotation，存放标注
│   │       ├── 00041000.txt
│   │       └── ...
│   └── ...
├── evaluation/                  # 官方评估脚本（Python 2！）
│   ├── script.py                # 主评估脚本（改自 ICDAR2015 RRC 脚本）
│   ├── rrc_evaluation_funcs.py  # 公共函数
│   ├── gt.zip                   # 测试集 500 张图的 ground truth（zip 内 500 个 txt）
│   └── readme.txt
├── tools/
│   ├── README.md                # 标注工具使用说明
│   ├── examples/                # 标注工具的示例清单（路径为占位符，需自行改）
│   └── PCBAnnotationTool/       # Qt 5.4.1 / C++ 编写的标注软件源码
└── fig/                         # README 用图（示例图对、缺陷数量统计图、检测效果图）
```

各 group 实测图片数（合计正好 1500 张待检图）：

| group | 00041 | 12000 | 12100 | 12300 | 13000 | 20085 | 44000 | 50600 | 77000 | 90100 | 92000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 张数 | 221 | 14 | 146 | 98 | 216 | 325 | 100 | 79 | 107 | 74 | 120 |

## 3. 数据格式详解

### 3.1 文件命名规则

以 ID `00041000` 为例，同一 ID 的三个文件构成一条完整样本：

| 文件 | 含义 |
|---|---|
| `group00041/00041/00041000_temp.jpg` | 模板图（无缺陷，供对齐/差分参考） |
| `group00041/00041/00041000_test.jpg` | 待检图（含缺陷，是检测网络的输入） |
| `group00041/00041_not/00041000.txt` | 该待检图的标注 |

规律：`{图片ID前5位}` 即 group 名后缀和子目录名（`00041`），标注目录是子目录名加 `_not`。

### 3.2 标注文件格式

每行一个缺陷框，5 个整数，空格分隔：

```
x1 y1 x2 y2 type
466 441 493 470 3
454 300 493 396 2
...
```

- `(x1, y1)`：左上角；`(x2, y2)`：右下角（**轴对齐框，两角点格式**）
- `type` 类别 ID 对照表（**0-background 未使用**）：

| ID | 类别 | 英文（评估脚本用名） | 实测框数 | 占比 |
|---|---|---|---|---|
| 1 | 开路 | `open` | 1,942 | 19.4% |
| 2 | 短路 | `short` | 1,506 | 15.0% |
| 3 | 鼠咬 | `mousebite` | 1,965 | 19.6% |
| 4 | 毛刺 | `spur` | 1,625 | 16.2% |
| 5 | 铜渣 | `copper` | 1,474 | 14.7% |
| 6 | 针孔 | `pin-hole` | 1,501 | 15.0% |
| — | **合计** | | **10,013** | 100% |

### 3.3 实测数据统计

- 标注文件 1,500 个，总框数 10,013（平均每图约 6.7 个框）
- 单张图缺陷数分布：**1 ~ 15 个**（官方 README 说"约 3~12 个"是近似口径；实测 5~8 个/图的样本占绝大多数，少数图只有 1 个框——转 YOLO/COCO 时按实际读到的为准即可）
- 坐标范围：0~640 以内（实测 min 5 / max 633），无需出界处理
- 图像为 JPEG 灰度（PIL 打开 mode=`L`），二值化后背景≈0（黑）、线路/焊盘≈255（白）；少数像素可能有中间值，加载时 `cv2.IMREAD_GRAYSCALE` 即可
- `trainval.txt` / `test.txt` 中**模板图与待检图路径成对出现**（`image_path annotation_path`，空格分隔），但只列了待检图（annotation）路径，模板图路径需按命名规则推导（见 3.4 的坑）

### 3.4 ⚠️ 使用本仓库必须知道的坑（实测发现）

1. **`trainval.txt` / `test.txt` 里的图片路径已失效（命名不匹配）**
   清单中写的是旧命名 `group20085/20085/20085000.jpg`，而仓库当前的实际文件是拆分后的
   `20085000_test.jpg` 与 `20085000_temp.jpg`，路径直接 `[ -f ]` 校验 100% 失败。
   需要做映射：
   ```python
   img_path  = line.split()[0].replace(".jpg", "_test.jpg")   # 待检图
   # 同目录下把 "_test.jpg" 换成 "_temp.jpg" 即模板图
   ann_path  = line.split()[1]   # 标注路径正常可用
   ```
2. **评估脚本是 Python 2 代码**：`script.py` 里有 `unicode('...')` 等 Py2 语法，
   在 Python 3 下直接跑会报 `NameError`。做 demo 时要么打个小补丁（`unicode`→`str`、print 兼容），
   要么自己按 3.5 的协议重写一个百行以内的评估器（推荐后者，顺便可输出 per-class AP）。
3. **`gt.zip` 与自测集的对应关系**：`evaluation/gt.zip` 内是测试集 500 张图的标注，
   文件名为纯 ID（如 `00041200.txt`），格式与 `*_not/*.txt` 相同。
4. 标注坐标以**待检图（test）**像素为基准，模板图上没有标注。
5. 类别字符串在评估协议里写作 `pin-hole`（带连字符），而标注 ID 是 `6`，两套表示法要建好映射。
6. `tools/examples/test.txt` 里的图片路径是 `/absolute/path/to/...` 占位符，直接跑标注工具会找不到文件，需要先改成真实路径。

### 3.5 官方评估协议

- **判定标准**：检测框与同类别 GT 框的 **IoU > 0.33** 才算命中（阈值远低于通用 0.5，
  因为 PCB 小缺陷框本身很小，0.5 过于苛刻）。
- **指标**：mAP 与 F-score（`F = 2PR/(P+R)`）并用。F-score 对阈值敏感，
  可自行调置信度阈值取最优；官方论文结果即 98.6% mAP / 98.2% F-score。
- **提交格式**：每张测试图一个 txt，每行
  ```
  x1,y1,x2,y2,confidence,type
  ```
  其中 `type` 是字符串（`open,short,mousebite,spur,copper,pin-hole`），**逗号分隔、无空格**；
  全部 txt 打包成 `res.zip`（不许有子目录），然后：
  ```bash
  python script.py -s=res.zip -g=gt.zip
  ```
- 评估脚本改自 ICDAR2015 Robust Reading Competition 官方脚本。

### 3.6 标注工具

`tools/PCBAnnotationTool` 是一个 Qt 5.4.1（C++）桌面程序，Windows 平台，
用 QtCreator 打开 `.pro` 编译即可。支持：打开清单批量浏览、框选标注/手动绘制新缺陷、
删除标注（只删记录不改图）。对 demo 项目作用不大，一般无需使用。

## 4. 与其他 PCB 缺陷数据集的对比（选型参考）

| 数据集 | 规模 | 特点 |
|---|---|---|
| **DeepPCB** | 1,500 对 / 6 类 / 约 1 万框 | 模板-待检图对，二值图，标注干净，最常用的 AOI 检测 benchmark |
| PKU-Market-PCB（北大） | 693 张 / 6 类 | 真实 JPG 彩图，含噪声，比 DeepPCB 更"脏"更接近实拍 |
| HUST-PCB | 约 300 张 | 类别多（约 13 类）但样本少 |

DeepPCB 图像是二值化后的理想化图像，训练出的模型对真实噪声鲁棒性有限——
demo 中可以演示"加噪声/形态学增强"来说明这一局限，是很好的加分点。

## 5. 二次开发指南：Python 开源 demo 路线

目标形态建议：`deep pcb-detector`，包含数据加载、格式转换、训练（YOLO）、评估、可视化/Gradio 演示。

### 5.1 推荐 demo 项目结构

```
deeppcb-demo/
├── data/                        # 指向 F:\data\PCB\DeepPCB-master 的软链或配置路径
├── deeppcb/
│   ├── dataset.py               # 清单解析 + 数据对加载（处理 3.4 的路径坑）
│   ├── convert_yolo.py          # 转YOLO格式（也可加 convert_coco.py）
│   ├── visualize.py             # 画框预览 / 结果可视化
│   └── eval_pcb.py              # Python3 版官方协议评估器（IoU=0.33, per-class AP）
├── train_yolo.py                # 训练入口（ultralytics YOLOv8/v11）
├── app.py                       # Gradio / Streamlit 演示页
├── requirements.txt
└── README.md
```

### 5.2 核心模块参考代码

**（1）数据集加载器**（处理命名坑 + 模板图配对）：

```python
from pathlib import Path
import cv2
import numpy as np

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]  # id-1 即索引

def load_split(root: str, split: str = "trainval"):
    """解析 trainval.txt / test.txt，返回样本列表。
    每个样本: dict(img_path, temp_path, ann_path, boxes=[x1,y1,x2,y2,cls])"""
    root = Path(root)
    list_file = root / "PCBData" / f"{split}.txt"
    samples = []
    for line in list_file.read_text().splitlines():
        if not line.strip():
            continue
        img_ref, ann_path = line.split()
        # 官方清单用旧命名 xxx.jpg，实际文件是 xxx_test.jpg
        img_path = root / "PCBData" / img_ref.replace(".jpg", "_test.jpg")
        temp_path = Path(str(img_path).replace("_test.jpg", "_temp.jpg"))
        boxes = np.loadtxt(root / "PCBData" / ann_path).reshape(-1, 5)
        samples.append(dict(img_path=img_path, temp_path=temp_path,
                            ann_path=root / "PCBData" / ann_path, boxes=boxes))
    return samples

def load_pair(sample):
    img = cv2.imread(str(sample["img_path"]), cv2.IMREAD_GRAYSCALE)   # 640x640 单通道
    temp = cv2.imread(str(sample["temp_path"]), cv2.IMREAD_GRAYSCALE)
    return img, temp
```

**（2）转 YOLO 格式**（YOLO 是最省事的训练路线；二值图转 3 通道即可喂给预训练权重）：

```python
def to_yolo_label(boxes, img_w=640, img_h=640):
    """[x1,y1,x2,y2,cls] -> YOLO txt 行列表：cls cx cy w h（归一化，cls 从 0 起）"""
    lines = []
    for x1, y1, x2, y2, c in boxes:
        cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
        lines.append(f"{int(c) - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
```

数据集配置 `pcb.yaml`：

```yaml
path: ./datasets/pcb
train: images/train
val: images/val
names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: copper
  5: pin-hole
```

训练（ultralytics）：

```bash
pip install ultralytics
yolo detect train data=pcb.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=32
```

**（3）Python3 版评估器**：按 3.5 的协议实现（同类别 IoU>0.33 匹配、按置信度排序算 AP、
输出 mAP 与最优阈值下的 F-score），约 150 行即可，建议作为 demo 的独立模块并写清与官方
`script.py` 的协议对应关系——这是与其他论文公平对比的关键。

### 5.3 Demo 亮点建议（开源项目差异化）

1. **模板差分可视化**：`cv2.absdiff(test, temp)` + 阈值化，直观展示"配对数据"的意义，
   也可作为传统方法 baseline 与深度模型对比。
2. **Gradio 演示页**：上传/选择测试图 → 模型预测 → 叠加显示预测框 vs GT 框。
3. **per-class 指标表 + 混淆矩阵**：6 类缺陷中 spur/copper 等小目标易混淆，值得展示。
4. **小目标问题讨论**：640×640 中多数框只有 20~50 像素宽，可在 README 里给出
   P2 检测头 / 更大输入分辨率 / mosaic 增强的消融建议。
5. **数据局限说明**：二值理想化图像 → 可演示对真实图加噪后的性能衰减，提示迁移局限。
6. 引用格式（README 里注明出处，尊重原作者）：

```bibtex
@article{ding2017on,
  title={On-line PCB Defect Detector On A New PCB Defect Dataset},
  author={Ding, Shangwei and Li, Mao and Yu, Zhengtao and others},
  journal={arXiv preprint arXiv:1902.10011},
  year={2019}
}
```

## 6. 快速验证清单（拿到数据后先跑一遍）

```python
# 1. 确认图像可读、尺寸/模式正确
from PIL import Image
im = Image.open(r"F:\data\PCB\DeepPCB-master\PCBData\group00041\00041\00041000_test.jpg")
print(im.size, im.mode)          # (640, 640) L

# 2. 画框预览一张图，确认坐标与缺陷位置对齐
import cv2, numpy as np
img = cv2.imread(r"...\00041000_test.jpg", 0)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x1, y1, x2, y2, c in np.loadtxt(r"...\00041_not\00041000.txt"):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
cv2.imwrite("preview.jpg", img)  # 打开检查框是否贴合缺陷

# 3. 核对训练/测试清单行数
#    trainval.txt 1000 行、test.txt 500 行（wc -l 少 1 是行尾无换行符所致）
```

预览框与缺陷贴合无误后，就可以放心进入格式转换与训练环节了。
