# SZ Classroom Predictor

---

## 简介 · Introduction

**中文：** **SZ Classroom Predictor** 是一款在 **Windows** 上运行的桌面应用程序，用于根据教室围护结构、遮阳与采光等参数，对能耗与光环境等指标进行快速预测，辅助方案比选与初步设计评估。**本软件的模型与结论仅适用于深圳地区教室场景**；用于其他城市或建筑类型时，结果仅供参考，不作为设计或审批依据。

**English:** **SZ Classroom Predictor** is a **Windows** desktop application for rapid prediction of energy and daylighting-related metrics from classroom envelope, shading, and daylighting parameters—supporting design comparison and early-stage evaluation. **The bundled models and outputs are intended for classroom buildings in Shenzhen only**; for other cities or building types, treat results as illustrative—not for design sign-off or compliance without local validation.

---

## 适用范围 · Applicability

**中文：**

- 本预测软件所依据的数据与训练设定针对 **深圳市** 相关教室设计语境；**仅推荐在深圳地区教室项目中作为辅助参考工具使用**。  
- 在深圳以外地区、或非典型教室形式下使用本软件时，预测值可能与实际偏差较大，**请谨慎解读**，必要时应结合当地规范、实测或专项模拟。

**English:**

- The models reflect data and assumptions aligned with **Shenzhen**-context classroom design; **the tool is meant as a supporting reference for classroom projects in Shenzhen**.  
- Outside Shenzhen or for non-standard classroom typologies, predictions may deviate substantially from reality—**interpret with caution** and validate against local codes, measurements, or dedicated simulation where decisions matter.

---

## 获取软件 · Obtaining the software

**中文：** 本仓库的 **Git 源码不等于可安装程序**。请从 **GitHub Releases** 下载已打包的发行版，在最新版本中获取 **`school_app_secure_portable.zip`**（或页面中说明的同名便携包）。

**[前往 Releases 下载页面](https://github.com/VincentWei2001/SZ-Classroom-Predictor/releases)**

**English:** The **Git repository is not the installable app**. Download the packaged release from **GitHub Releases** and get **`school_app_secure_portable.zip`** (or the portable package named on that page) from the latest release.

**[Go to Releases](https://github.com/VincentWei2001/SZ-Classroom-Predictor/releases)**

---

## 安装与启动 · Installation and launch

**中文：**

1. 将 **ZIP** 解压到本地文件夹（建议使用字母、数字与常见符号路径，避免路径仅含特殊字符）。
2. 进入解压目录，双击 **`school_app.exe`** 启动。
3. **请勿**单独移动 `school_app.exe`；须与 **`_internal`** 保持原有相对位置，否则可能无法运行。  
   本版为便携包，**无需安装 Python**。

**English:**

1. Extract the **ZIP** to a local folder (prefer paths with letters, numbers, and common symbols).
2. Open the folder and double-click **`school_app.exe`**.
3. **Do not** move `school_app.exe` alone; keep **`_internal`** next to it as shipped.  
   This is a portable build—**no Python installation** is required.

---

## 系统要求 · System requirements

| 中文 | English |
|------|---------|
| 操作系统：Windows 10 / 11（64 位） | OS: Windows 10 / 11 (64-bit) |
| 建议分辨率不低于 1280×720 | Display: 1280×720 or higher recommended |

---

## 使用说明 · Usage

**中文：** 启动后在图形界面中选择朝向、遮阳模式等，调整参数后进行预测并查看结果；具体以软件界面为准。

**English:** After launch, choose orientation, shading mode, and other inputs in the GUI, then run predictions and review results; follow on-screen guidance for details.

---

## 常见问题 · FAQ

**中文：**

- **无法启动或缺文件：** 确认已完整解压，且 `school_app.exe` 与 `_internal` 未被安全软件隔离或删除。  
- **安全软件告警：** 未签名 exe 可能被提示风险；若文件来自本仓库 **Releases** 官方附件，可按环境添加信任或咨询信息化部门。  
- **是否需要克隆仓库：** 仅使用软件时，只需下载 Releases 压缩包；仓库用于源码公开与交流。

**English:**

- **Won’t start or missing files:** Ensure full extraction and that `school_app.exe` and `_internal` were not quarantined or deleted.  
- **Antivirus warnings:** Unsigned executables may be flagged; if you downloaded from official **Releases** assets, add an exception if appropriate or consult your IT policy.  
- **Do I need to clone the repo?** For normal use, download the release ZIP only; the repo is for source code and discussion.

---

## 开源与源码 · Open source

**中文：** 源代码托管于 GitHub；主逻辑见 **`classroom_predictor_app.ipynb`** 及导出脚本，构建与模型打包脚本在仓库根目录。自行构建需要 Python 与相应依赖环境。

**English:** Source is on GitHub; core logic is in **`classroom_predictor_app.ipynb`** and exported scripts, with build scripts at the repository root. Building from source requires Python and the documented dependencies.

---

## 许可协议 · License

**中文：** 若仓库未包含 `LICENSE` 文件，则权利保留；公开发布时建议由权利方补充明确许可条款。

**English:** If no `LICENSE` file is present, all rights are reserved; for public distribution, the rights holder should add explicit license terms.
