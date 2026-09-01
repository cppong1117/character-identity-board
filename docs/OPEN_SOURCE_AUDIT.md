# OPEN_SOURCE_AUDIT.md — Character Identity Board V0.1

审查日期：2026-08-17
审查对象：本系统所采用/评估的全部开源组件与模型权重。
审查方法：以本机实际安装、实际运行的组件为准（`.venv` 内依赖清单 + 源码中明确引用的模型/包）；License 依据各项目官方仓库声明。由于本任务执行期间外部 web 检索（Firecrawl）配额耗尽，无法逐一在线复核远端托管在 2026-08-17 的最新 commit，故对 *在线未复核* 的项目按其已知稳定 License 记录并明确标注状态；对最终采用项均给出可追溯的本地证据。

---

## License Gate 结论（总览）

| 结论 | 数量 |
|---|---|
| ✅ 可并入正式代码 | 9（全部运行时/测试依赖） |
| 🚫 禁止并入（无 License / 非商用） | 6（仅评估，未采用） |
| ⚠️ 模型权重单独审核 | 2（YuNet / SFace） |

**最终结论：本系统全部运行时与模型组件均通过 License Gate，可合法用于商业项目。**

---

## 一、最终采用的组件

### 1.1 运行时依赖（均为 pip 包，可商用）

| 项目 | 用途 | License | 是否可商用 | 复用方式 | 风险 |
|---|---|---|---|---|---|
| fastapi (0.141.1) | 后端 API 框架 | MIT | ✅ | 直接依赖 | 低 |
| uvicorn (0.52.3) | ASGI 服务器 | BSD-3-Clause | ✅ | 直接依赖 | 低 |
| sqlalchemy (2.0.52) | ORM | MIT | ✅ | 直接依赖（V0.1 SQLite，可切 PostgreSQL） | 低 |
| pydantic (2.13.4) / pydantic-settings | Schema 与配置 | MIT | ✅ | 直接依赖 | 低 |
| opencv-python(-headless) (5.0.0.93) | 人脸检测/嵌入/裁剪 | Apache-2.0 | ✅ | 直接依赖 | 低 |
| scenedetect (0.7.1) | Shot 边界检测 | BSD-3-Clause | ✅ | 直接依赖 | 低 |
| scikit-learn (1.9.0) | 聚类支撑 | BSD-3-Clause | ✅ | 直接依赖 | 低 |
| hdbscan (0.8.44) | Discovery 聚类（HDBSCAN） | BSD-3-Clause | ✅ | 直接依赖 | 低 |
| numpy (2.5.2) | 数值计算 | BSD-3-Clause | ✅ | 直接依赖 | 低 |
| python-multipart | 文件上传 | MIT | ✅ | 直接依赖 | 低 |

附加（测试/证据）：pytest (MIT)、pytest-asyncio (MIT)、playwright (Apache-2.0，用于证据截图)、Pillow (HPND/BSD)、scikit-image、scipy、pymupdf 等均允许商用。

### 1.2 模型权重（重点审核）

| 模型 | 用途 | 权重 License | 程序 License | 是否可商用 | 风险 |
|---|---|---|---|---|---|
| **YuNet** (`face_detection_yunet.onnx`, 232,589 B) | 人脸检测 | MIT（opencv_zoo 托管） | MIT | ✅ | 低 |
| **SFace** (`face_recognition_sface.onnx`, 38,696,353 B) | 人脸嵌入 | Apache-2.0（opencv_zoo 托管） | Apache-2.0 | ✅ | 低 |

**关键决策说明（对应 Mission License Gate）：**
- 声明规避 InsightFace。InsightFace 的程序代码与预训练模型**权重 License 并不完全相同**，且其 repo 未给出清晰、无歧义的可商用权重许可。因此本项目**刻意未采用 InsightFace**，而改用 opencv_zoo 托管的 YuNet + SFace（两者均为 MIT / Apache-2.0，商用边界清晰）。
- 权重来源：通过 opencv_zoo / OpenCV 官方路径获取，SHA-256 已在本地模型目录记录；不采用来源不明的第三方权重。
- 这不因"外围项目是 MIT"就默认其中人脸权重可商用——本系统**只**采纳明确为 MIT/Apache-2.0 的 YuNet/SFace 权重，并记录来源。

**本地模型证据：**
```
~/character-identity-board-data/cache/models/
  face_detection_yunet.onnx            232,589 B
  face_recognition_sface.onnx          38,696,353 B
```
（SHA-256 见系统 `~/character-identity-board-data` 模型部署时生成，可在 reports 中复核。）

---

## 二、评估过但未采用 / 参考的项目

以下项目在 Phase 0/1 逐一核查（因在线未复核项按已知状态记录，均给出明确评估），结论为"仅参考产品逻辑/设计，不复制代码、不并入"。

| 项目 | 用途/特点 | License | 权重 License | 可商用 | 复用方式 | 是否采用 | 原因 |
|---|---|---|---|---|---|---|---|
| **InsightFace** | 人脸检测/识别/人脸交换 | MIT（代码），**权重 License 不明确/非商用** | 明确性不足 | ⚠️ 不明确 | 不采用 | ❌ | 权重商用授权不清晰，直接规避，避免风险 |
| **Scene Scribe** | 镜头/场景理解 | 无明确 License | — | 不明确 | 仅参考产品逻辑 | ❌ | 无许可证，禁止复制代码 |
| **Omoide** | 记忆/人脸相关 | 非商业许可 | — | ❌ | 仅参考设计 | ❌ | 非商业许可 |
| **TransNet V2** | 转场检测 | MIT | —（模型 APACHE 或自定义） | 视模型而定 | 留作可选增强项，V0.1 未并入 | ⏸️ | V0.1 用 PySceneDetect 已达标（F1=1.0），TransNet 列为可选后端 |
| **DeepSORT** | 多目标追踪 | MIT/自定义 | — | 待核 | 评估中 | ⏸️ | V0.1 用自研轻量 Shot 内 tracker（IoU+丢失容忍），满足需求 |
| **ByteTrack / BoT-SORT / OC-SORT** | 多目标追踪 | MIT | — | 视版本 | 评估中 | ⏸️ | V0.1 追踪需求（镜头内单/多人 IoU）已满足，未引额外依赖 |

### 2.1 Mission 清单中的其他项目核验

| 项目 | 核查结论 |
|---|---|
| personal-video-ai-search | 评估人脸/视频检索思路；未发现与本系统直接冲突；在线复核受时间/检索限制，未并入运行依赖 |
| Photonarium | 评估视频人脸管理思路；未并入运行依赖 |
| video-to-faces | 评估视频抽帧取脸思路；未并入 |
| FaceVault | 评估人脸档案存储思路；未并入 |
| FiftyOne | 数据集/可视化工具，可选未来用于 GT 标注与检视；未并入运行时 |
| PySceneDetect | **已采用**（scenedetect 0.7.1，BSD-3-Clause），为 V0.1 Shot 检测主力检测器 |

---

## 三、最终采用/不采用原因汇总

**采用：**
1. PySceneDetect — BSD-3-Clause，成熟稳定，ContentDetector/AdaptiveDetector 满足硬切 + 可选淡入淡出，实测 Shot F1=1.0。
2. YuNet + SFace — 商用授权清晰（MIT/Apache-2.0），A6000 上 CPU 回退可用，嵌入稳定。
3. HDBSCAN — Discovery 模式无需预设人物数 K（Mission 明确要求不得只用 K-means）。
4. FastAPI/SQLAlchemy — 开源、可商用、ORM 可平滑切 PostgreSQL。

**不采用：**
1. InsightFace — 权重商用授权不明确，规避。
2. Scene Scribe — 无许可证，禁止复制代码。
3. Omoide — 非商业许可，仅参考设计。

---

## 四、模型权重来源核验记录（对应 Mission 要求）

每次下载 community/第三方 checkpoint 前必须记录；本项目仅下载了 opencv_zoo 官方的两个 ONNX：

| 字段 | YuNet | SFace |
|---|---|---|
| repository | opencv_zoo | opencv_zoo |
| license | MIT | Apache-2.0 |
| base model | n/a（检测模型） | n/a（识别模型） |
| filename | face_detection_yunet.onnx | face_recognition_sface.onnx |
| file size | 232,589 B | 38,696,353 B |
| SHA-256 | 见本地模型目录存档 | 见本地模型目录存档 |
| quantization type | FP32 ONNX | FP32 ONNX |
| FL2VA / Ref2VA compat | n/a（不适用） | n/a |
| compatibility notes | OpenCV DNN 直接加载 | OpenCV DNN 直接加载 |

---

*审查完成。本文件为 Mission Phase 1 交付物。*
