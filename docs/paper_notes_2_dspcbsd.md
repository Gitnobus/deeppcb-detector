# 论文解读（二）· DsPCBSD+ 数据集

> **A dataset for deep learning based detection of printed circuit board surface defect**
> Shengping Lv（通讯作者）, Bin Ouyang, Zhihua Deng, Tairan Liang, Shixin Jiang, Kaibin Zhang, Jianyu Chen, Zhuohui Li
> （华南农业大学 工程学院；Guangzhou FastPrint Technology Co., Ltd 产线；CEPREI 赛宝实验室）
> *Scientific Data* (Nature 旗下数据期刊), 2024, 11:811 — DOI: [10.1038/s41597-024-03656-8](https://doi.org/10.1038/s41597-024-03656-8)
> 数据仓库：Figshare — DOI: [10.6084/m9.figshare.24970329](https://doi.org/10.6084/m9.figshare.24970329)
> 本地文件：`DeepPCB_detect/references/A dataset for deep learning based detection of printed circuit board surface defect.pdf`
>
> 这是一篇**数据描述（Data Descriptor）论文**：不提出新模型，核心贡献是发布一个
> 真实工业产线的 PCB 表面缺陷数据集 **DsPCBSD+**。Open Access (CC BY 4.0)。

---

## 0. 一页速览（TL;DR）

| 项 | 内容 |
|---|---|
| 数据来源 | 广州快联（FastPrint）真实 PCB 产线的 AOI 设备（AGLE'OL AOI-100 V8），**真实缺陷**而非人工合成 |
| 规模 | 10,259 张 226×226 JPG 图像，**20,276 个人工标注框**，9 类缺陷 |
| 格式 | 官方直接提供 **YOLO 和 COCO** 两种格式（另附 VOC→YOLO→COCO 转换脚本链接与去重脚本 Hash.py） |
| 划分 | 8 : 2 随机划分（训练 8,208 图 / 16,184 框；验证 2,051 图 / 4,092 框） |
| 基准成绩 | Co-DETR mAP50 ≈ 0.848，YOLOv6-L6 ≈ 0.851 |
| 与 DeepPCB 的关系 | 针对 DeepPCB 等旧数据集"人工合成、类内多样性不足、类别覆盖不全"的缺陷，用真实数据补位——是 demo 的天然"第二数据集/泛化性对照" |

## 1. 写作动机：现有数据集的四大缺陷

论文用一整节逐条批评现有公开数据集（含 DeepPCB、PKU-Market-PCB 等），这是理解其定位的关键：

1. **人工合成/增强占比过高**：DeepPCB、PKU-Market-PCB 主要靠人工合成；Ding et al.、Hu et al.、Liao et al. 等的数据集大多由少量原始缺陷增强生成 → **类内多样性（intra-class variability）严重不足**，与真实产线缺陷分布差距大；
2. **类别覆盖不全或划分粗糙**：有的数据集缺陷不分类或只分 2 类；最多也只有 5~6 类、约 2,000 个缺陷 → 需要"更全覆盖 + 更细分类"；
3. **先增强后划分导致数据泄漏**：不少数据集在增强之后才切分 train/val，验证集里充满与训练集几乎相同的样本 → 无法检验真实泛化能力；
4. **AOI 滑窗裁剪引出的脏数据无人处理**：AOI/AVI 滑窗会产出无缺陷图、重复缺陷图、缺陷不完整图，现有数据集未说明如何清洗；另外只凭 2D 目视无法判定的缺陷（需模板匹配）、类别不均衡问题也缺乏处理指引；且这些数据集多数不公开。

> 解读：这四条几乎每条都点名了 DeepPCB 的痛处。做 demo 的 README/讨论章节时，
> 这一段非常适合作为"数据集局限性"的论据来源。

## 2. 数据集构建（Methods）

### 2.1 图像采集

- 设备：AGLE'OL AOI-100 V8 AOI 机，多组可控 LED 点光源 + **16K 高分辨率线扫系统**（上下各 4 台相机，拍摄 PCB 两面）；
- 来源：广州快联产线的**内层/外层板蚀刻后**的真实缺陷图像；
- 原始取出 32,259 张，经清洗后保留 10,259 张，每张 226×226 JPG。

### 2.2 缺陷分类体系（9 类，4 大成因）

先按**成因**分 4 大类，再按**位置 + 形态**细分 9 类：

| 成因大类 | 细分（缩写） | 描述 |
|---|---|---|
| 铜残留（Copper residue） | **SH** Short（短路） | 残铜导致不同导体间意外连接；多发生于线与线之间，也见于铜面与线、铜面之间 |
| | **SP** Spur（毛刺） | 导体边缘不规则凸起，尖锐刺状 |
| | **SC** Spurious copper（杂铜） | 基材/孔/铜面上的多余铜 |
| 铜缺失（Copper deficiency） | **OP** Open（开路） | 导体内传导路径中断 |
| | **MB** Mouse bite（鼠咬） | 导体边缘小范围凹陷/裂口 |
| | **HB** Hole breakout（孔破） | 孔中心明显偏离焊盘，孔边缘显著缺铜 |
| 导体划伤（Conductor scratch） | **CS** Conductor scratch（导体划伤） | 线/焊盘/铜面上的线状或多线状划痕 |
| 外来物（Foreign object） | **CFO** Conductor foreign object（导体异物） | 导体上的杂质/气泡/污垢/油污/氧化物等 |
| | **BMFO** Base material foreign object（基材异物） | 基材上的气泡/碎屑/颗粒等；孔内异物按形态归入此类 |

分类设计的两个特点：① 明确给出"成因 + 位置 + 形态"的**分类标准**（旧数据集普遍缺失，导致 Spur 与 Spurious copper 这类易混类别标注不一致）；② 特意纳入旧数据集忽略的 HB / CS / CFO / BMFO——论文统计显示这三类在实际产线中占比很大。

### 2.3 数据清洗与均衡（对应动机第 4 条）

按四步过滤原始 32,259 张图：
1. **无缺陷图**：人工筛除（缺陷过小目视不可辨的也一并剔除）；
2. **重复缺陷图**：AOI 对同一缺陷多次拍摄产生部分重叠图 → 用**哈希值匹配**去重（随数据集附 `Hash.py`），再人工保留缺陷占比最高的一张；
3. **缺陷不完整图**：缺陷边界不完整（如 Open 只拍到断线一端、Short 只含导体一侧）→ 剔除；
4. **其他类别图**：仅靠模板匹配才能发现、2D 目视不可判定的缺陷（线宽/间距超差、盲孔凹陷、漏钻孔等）→ 剔除。

**类别均衡处理**（有产线逻辑，很值得借鉴）：清洗后 OP/SH 极少而 CFO/BMFO 极多。
由于 OP/SH 会直接导致 PCB 报废而 CFO/BMFO 通常不会，**不能让模型偏向多数类**——
因此 OP/SH 全部保留，其余类别按统计结果随机抽样补齐，使各缺陷类型分布尽量均衡。

### 2.4 标注与划分

- 工具：LabelImg（VOC 格式起标），每框标 9 类缩写；一图多类缺陷时逐个标注；
- 最终 10,259 图 / 20,276 框；VOC → 脚本转 YOLO / COCO（附 RapidAI 的 VOC2YOLO、YOLO2COCO 链接）；
- **先标注、后 8:2 随机划分**（明确避免"先增强后划分"的泄漏问题）：train 8,208 图 / 16,184 框，val 2,051 图 / 4,092 框。

### 2.5 尺寸分布（Table 1，小目标问题一目了然）

按 COCO 标准（<32×32 小 / 32~96 中 / >96 大）：

| 类别 | 小 | 中 | 大 | 合计 |
|---|---|---|---|---|
| SH | 710 | 205 | 0 | 915 |
| SP | 4,469 | 115 | 0 | 4,584 |
| SC | 1,352 | 231 | 10 | 1,593 |
| OP | 1,406 | 361 | 3 | 1,770 |
| MB | 2,421 | 108 | 0 | 2,529 |
| HB | 35 | 2,848 | 0 | 2,883 |
| CS | 734 | 1,043 | 713 | 2,490 |
| CFO | 1,140 | 582 | 110 | 1,832 |
| BMFO | 1,308 | 304 | 68 | 1,680 |
| **合计** | **13,575** | **5,797** | **904** | **20,276** |

**约 67% 的缺陷是小目标**；SP、MB 几乎全是小目标；CS/CFO/BMFO 尺寸方差大、类内差异也大。

## 3. 技术验证（Technical Validation）

### 3.1 人工核验

5 位 PCB 行业专家逐图核验标注；对易混情形（跨元素缺陷、同类相邻、多框重叠）采用**集体讨论**定标签，综合考虑缺陷对性能的影响程度、位置占比、可见性。

### 3.2 基准实验

两个 COCO 榜上 SOTA 模型（默认超参 + 少量适配，RTX 3090）：

| 模型 | 输入尺寸 | 训练耗时 | AP50 | AP75 | AP50:95 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|---|---|
| Co-DETR | 1333×800 | ~69 min | 0.848 | 0.490 | 0.492 | 0.425 | 0.554 | 0.671 |
| YOLOv6-L6 | 1280×1280 | ~721 min | 0.851 | 0.525 | 0.514 | 0.405 | 0.597 | 0.681 |

per-class（IoU=0.50）要点：
- 最容易的类：**HB**（AP≈0.97，孔破位置固定、形态独特）、OP、SH；
- 最难的类：**CS**（AP≈0.73，尺寸方差大，小划痕易漏检）、CFO（颜色/形态多样，易被判成背景）、SP（极小、不显眼）；
- 典型混淆：**SC → SP**（SC 紧邻导体时形态与 SP 相似，错误率约 0.06~0.07）；
- 小目标 AP（AP_S 0.40~0.43）显著低于大目标（0.67~0.68），再次印证 PCB 缺陷检测=小目标检测。

### 3.3 五折交叉验证

训练集再分 4 份轮换验证，Co-DETR5 AP50=0.840、YOLOv6-L65=0.837，与原划分几乎一致 → 各折均能代表全体样本空间，划分稳健。

## 4. 数据记录（Data Records）

```
Figshare: 10.6084/m9.figshare.24970329
├── Data_YOLO/
│   ├── images/{train, val}/     # 226×226 JPG
│   └── labels/{train, val}/     # cls cx cy w h（归一化，YOLO txt 格式）
└── Data_COCO/
    ├── train2017/  val2017/
    └── annotations/instances_{train,val}2017.json
```

## 5. Usage Notes：与其他数据集的类别对照（Table 5 摘录）

| 数据集 | 类别 | 年份 | 数据性质 |
|---|---|---|---|
| **DeepPCB** | Open, Short, Mousebite, Spur, Copper, Pin-hole | 2019 | 真实扫描图像 + 人工合成缺陷，二值图 |
| PKU-Market-PCB 及扩展 | Missing hole, Mouse bite, Open, Short, Spur, Spurious copper | 2019 | 10 块实板 + 旋转增强 |
| MeiweiPCB | 单一类型 | 2021 | 线扫相机裁剪，含像素级 mask |
| Hu et al. | 含 Solder ball 的 6 类 | 2020 | 增强至 12,000 张 |
| Liao et al. | 6 类（凹凸、杂物、划痕…） | 2021 | 增强至 19,029 张 |
| **DsPCBSD+** | SH, SP, SC, OP, MB, HB, CS, CFO, BMFO | 2024 | **全真实 AOI 产线缺陷** |

作者声明自己的局限（论文 Usage Notes 末尾，做 demo 引用时注意）：
1. AOI 相机无 3D 深度信息，凸起/凹陷类缺陷无法识别（只覆盖 2D 表面缺陷）；
2. 只包含**蚀刻后内/外层板**的缺陷，不含阻焊（solder mask）之后的工序；
3. 图像是从整板裁出的局部区域，实际部署需要把局部图拼回整板图定位。

## 6. 对我们 demo 项目的启示

1. **作为泛化性对照**：demo 可以主打 DeepPCB（二值、干净、易出效果），在讨论/实验章节用 DsPCBSD+ 做"真实数据上的表现对照"——从 DeepPCB 训练的模型在真实图像上性能会明显下降，这是很好的说明性实验；也可直接把 DsPCBSD+ 作为 demo 的第二数据集（官方 YOLO 格式即插即用，接入 ultralytics 只需写一个 `pcb.yaml`）；
2. **类别映射**：DsPCBSD+ 的 SH/SP/SC/OP/MB 与 DeepPCB 的 6 类有交集（无 pin-hole、多 HB/CS/CFO/BMFO），跨数据集实验时需明确映射或取公共子集；
3. **工程实践可借鉴**：哈希去重、缺陷不完整图剔除、"会报废的类全保留"的均衡策略、先标注后划分——都是数据集工程的模板做法；
4. **评估口径不同**：DsPCBSD+ 用 COCO 标准（AP50:95、AP_S/M/L），DeepPCB 用宽松的同类 IoU>0.33——两套数字不可直接互比，demo 若同时报告两个数据集要分开说明；
5. **引用**：

```bibtex
@article{lv2024dspcbsd,
  title={A dataset for deep learning based detection of printed circuit board surface defect},
  author={Lv, Shengping and Ouyang, Bin and Deng, Zhihua and Liang, Tairan and Jiang, Shixin and Zhang, Kaibin and Chen, Jianyu and Li, Zhuohui},
  journal={Scientific Data},
  volume={11},
  pages={811},
  year={2024},
  doi={10.1038/s41597-024-03656-8}
}
```
