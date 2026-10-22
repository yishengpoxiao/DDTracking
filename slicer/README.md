# 在 3D Slicer 中使用 DDTracking

本仓库包含一个 3D Slicer 模块，可从图形界面启动纤维追踪。Slicer 负责界面和加载原生空间结果；配准、特征准备和模型推理由独立的 Python 环境执行，避免把模型依赖安装进 Slicer 自带的 Python。

## 版本兼容范围与源码安装

DDTracking 是 Python scripted module，没有需要针对 Slicer 版本编译的 C++ 扩展，也没有限定 5.2.2 的版本号判断。同一份源码以使用 **Python 3 和 Qt 5** 的 Slicer **4.11 及 5.x** 为兼容目标，通过稳定的 `ScriptedLoadableModule`、Qt 和 MRML/VTK 接口运行。SimpleITK、DIPY 和 PyTorch 在独立后端中运行，不受 Slicer 内置 Python 版本影响。Python 2 时代的 Slicer 不支持；尚未发布或采用不同 Python/Qt 主版本的构建需要重新验证。

已在 Linux 上实测 **4.11.20210226、5.2.2 和 5.12.4**，详见下方验证表；其他发行版和平台尚未逐一验证。安装方式遵循 Slicer 的源码模块工作流：将含有模块 `.py` 文件的目录添加到 Additional module paths。更多背景见官方 [Slicer 4.11 设置文档](https://slicer.readthedocs.io/en/v4.11/user_guide/settings.html)、[当前设置文档](https://slicer.readthedocs.io/en/latest/user_guide/settings.html) 和 [Python FAQ](https://slicer.readthedocs.io/en/5.6/developer_guide/python_faq.html)。

本指南描述从源码运行模块，不是 Extensions Manager 安装包，也不要求使用 Slicer SDK 构建二进制。请保持仓库完整，以便模块从 `slicer/DDTracking` 相对定位 `slicer_backend/run.py` 和 `resources/MNI_FA_template.nii.gz`。

## 环境要求

- 已安装上述兼容范围内的 3D Slicer，并能访问 `slicer/DDTracking` 模块目录。
- 为后端准备独立的 **Python 3.10** 环境，并安装 [`../requirements-slicer-2026-10-22.txt`](../requirements-slicer-2026-10-22.txt) 中的依赖。这个版本建议只针对后端，不要求更改 Slicer 内置 Python。
- **PyTorch 2.4 或以上**需单独安装。CPU 用户请从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择 CPU wheel；CUDA 用户根据操作系统和 CUDA 环境选择合适的 wheel。所选 PyTorch 必须能在后端 Python 环境中运行；Slicer 本身不需要安装 PyTorch。
- 与数据匹配的 DDTracking 模型检查点。项目在 [GitHub v0.1 release](https://github.com/yishengpoxiao/DDTracking/releases/tag/v0.1) 提供模型权重（压缩包约 154 MB）。下载并解压后，在模块中选择对应的检查点文件。请保留权重的来源信息及其适用条款。

先获取完整仓库，也可从 GitHub 使用 **Code → Download ZIP** 后解压：

```bash
git clone https://github.com/yishengpoxiao/DDTracking.git
cd DDTracking
```

从仓库根目录创建后端环境：

```bash
conda create -n ddtracking-slicer python=3.10 pip -y
conda activate ddtracking-slicer
python -m pip install -r requirements-slicer-2026-10-22.txt
# 再按上面的官方选择器安装 CPU 或适用于本机的 CUDA 版 PyTorch。
python slicer_backend/run.py --check-dependencies
python -c "import sys; print(sys.executable)"
```

依赖文件没有列出 PyTorch，因为正确的安装包取决于运行设备。仅运行 Slicer 模块时不需要安装完整的 `requirements.yaml`；该文件还包含训练和其他用途的包。

## 在 Slicer 中安装模块

1. 克隆或下载本仓库，并将其放在运行 Slicer 期间保持可用的位置。
2. 在 Slicer 中打开 **Edit → Application Settings → Modules → Additional module paths**。不同操作系统可能把设置入口放在应用菜单或称为 Preferences；添加下表中的**模块目录**：

   | 系统 | 添加到 Additional module paths 的目录 |
   | --- | --- |
   | Windows | `C:\path\to\DDTracking\slicer\DDTracking` |
   | Linux | `/path/to/DDTracking/slicer/DDTracking` |
   | macOS | `/path/to/DDTracking/slicer/DDTracking` |

   这里要添加的是直接包含 `DDTracking.py` 的 `slicer/DDTracking` 文件夹，**不是**仓库根目录，也不是 `slicer` 父目录。Slicer 会在启动时扫描额外模块目录；详见对应版本的官方设置文档。
3. 重启 Slicer，在模块选择器中打开 **Diffusion → DDTracking**。升级或另装一个 Slicer 版本后，检查那个版本的模块路径；如果没有继承该设置，再添加同一目录。
4. 在 **External Python** 中选择后端环境的 Python **可执行文件**，不要选择环境目录：Linux/macOS conda 环境通常是 `<env>/bin/python`，Windows 通常是 `<env>\python.exe`。若不确定具体路径，在该后端环境激活后运行 `python -c "import sys; print(sys.executable)"`，再把输出路径填入。请勿用 Slicer Python Console 中的 `sys.executable`，它指向 Slicer 自带 Python。Slicer 与后端各自使用自己的 Python；无需在 Slicer 内安装 PyTorch、DIPY 或后端 requirements。
5. 选择必需输入文件和一个已经存在的输出目录。建议先点 **Register FA only**，检查配准后的 FA 与模板；确认配准 QC 后，再选择匹配的模型检查点并点 **Run tracking**。FA-only 模式不需要模型检查点。

源码目录示例：

```text
DDTracking/                 # 保持整个仓库布局，模块通过它定位后端和默认模板
├── resources/
├── slicer_backend/
└── slicer/
    └── DDTracking/         # 添加到 Slicer 的 Additional module paths
        └── DDTracking.py
```

模块会异步启动后端，因此运行时 Slicer 仍可响应。日志区域显示后端输出和错误；点击 **Cancel** 可请求终止运行进程。成功完成后，模块会把原生空间纤维加载到当前场景中。结果图像和变换文件保存在输出目录；勾选 QC 选项还可加载配准后的 FA 和 MNI 空间纤维。

路径含空格时，在文件选择框里使用完整路径即可，**不要**在路径两侧手动加引号。模块通过 Qt `QProcess` 将程序和参数分开传递。若提示后端不存在，请确认没有只复制 `slicer/DDTracking` 子目录，并且仓库根目录下仍有 `slicer_backend/run.py` 与 `resources/MNI_FA_template.nii.gz`。

### 常见安装问题

- **模块列表里没有 DDTracking：** 检查添加的是包含 `DDTracking.py` 的 `slicer/DDTracking` 目录，重启 Slicer，并在 Modules 设置页查看模块是否被忽略或加载失败。也可查看 Slicer 启动日志中的 Python 错误。
- **外部进程无法启动或依赖检查失败：** `External Python` 必须是后端环境中的可执行文件。先在终端对该环境运行 `python slicer_backend/run.py --check-dependencies`；CPU/CUDA 版 PyTorch 必须安装在同一个环境中。
- **模板或后端文件找不到：** 保留完整仓库结构，并从该源码 checkout 加载模块。不要将模块文件单独复制到一个脱离仓库的位置。
- **更换或升级 Slicer 后模块消失：** 在新安装的 Slicer 设置里重新添加模块目录并重启。源码模块无版本专用二进制，但仍需对目标 Slicer 版本进行本机验证。

## 输入数据

必须提供单被试 DWI NIfTI 图像（`.nii` 或 `.nii.gz`）、与其匹配的 FSL 格式 `.bval` 和 `.bvec` 文本文件，以及处于同一原生空间的脑掩膜。b 值和 b 向量的数量必须与 DWI 第四维的体积数一致。b 向量文件可以是 3 行或 3 列；方向需符合该图像对应的 FSL 坐标约定。请确认图像、梯度和掩膜来自同一次采集且几何位置一致。

白质掩膜和预先计算的 FA 图像是可选项。若未提供 FA，后端会使用 b=0 体积以及所选的单一扩散壳层拟合张量并计算 FA。若未提供白质掩膜，追踪器使用 **GA>0.18** 生成种子，并以 **FA>0.15** 作为终止边界；提供白质掩膜时则使用该掩膜播种并作为边界。无论是否提供上述可选项，都必须提供脑掩膜。

默认选择 **1000 s/mm²** 壳层，容差为 **±100 s/mm²**；用于张量拟合的 b=0 体积默认满足 b≤**50 s/mm²**。若数据中没有 b=0 体积，或所选壳层没有扩散方向，则默认设置无法处理。运行前请核对采集数据中的 b 值。

默认模型使用 SH 阶数 6，需要所选壳层中至少 **28 个能张成该 SH 特征空间的独立方向**；仅满足方向数量但采样方向退化也会被拒绝。FA 的张量拟合至少需要 6 个可确定张量的扩散方向，因此只做配准时对方向采样的要求较低。

## 默认追踪参数

默认每个符合条件的体素生成 **1 个种子**，步长为 **0.5 个体素**，最大转角 **45°**，纤维长度范围 **40–200 mm**，每批处理 **1024 个种子**。设备默认是 CPU；如果后端环境安装了可用的 CUDA 版 PyTorch，可在设备字段中选择 `cuda:0`。显存不足时可调小批次大小；CPU 运行可能耗时较长。开始运行前可在模块中修改这些参数。

## 配准和输出

默认模板为仓库内的 `resources/MNI_FA_template.nii.gz`。后端通过 SimpleITK 估计原生 FA 到模板的刚性配准变换，将 DWI 和掩膜重采样到模板空间，并按同一刚性变换旋转扩散梯度。随后在模板空间执行追踪，再将纤维映射回原生 DWI 空间，以便在 Slicer 中显示和用于后续分析。

每次运行都会在指定输出目录下创建独立的子目录，其中包含带日期的运行配置、后端日志、结果清单、原生空间和 MNI 空间的 VTP 与 TRK 纤维、原生 FA、配准后的 FA、模板引用以及原生到 MNI 的变换。VTP 使用 RAS 坐标并包含 `SPACE=RAS` 字段；Slicer 模块将其点坐标直接作为 RAS 加载。清单提供 `native_vtp`、`mni_vtp`、`native_trk`、`mni_trk`、`native_fa`、`registered_fa`、`template` 和 `native_to_mni_transform` 路径。结果准备完成后，后端会输出 `DDTRACKING_RESULT=/path/to/result.json` 行。

请将配准后的 FA 与模板对照检查，并在解释追踪结果前检查原生空间纤维。这里的刚性配准流程是新实现，与仓库原先基于 FSL 和 ANTs 的预处理在数值上并不完全相同；目前没有在真实数据上测量二者的等价性。配准错误、梯度坐标约定错误，或检查点训练时使用了不同预处理，都可能影响纤维结果。

## 后端命令行

图形界面调用 `slicer_backend/run.py`。也可从仓库根目录直接运行后端以排查问题：

```bash
python slicer_backend/run.py \
  --dwi /data/sub-01_dwi.nii.gz \
  --bvals /data/sub-01_dwi.bval \
  --bvecs /data/sub-01_dwi.bvec \
  --brain-mask /data/sub-01_brain-mask.nii.gz \
  --model /models/ddtracking.pt \
  --output-dir /data/ddtracking-output \
  --device cpu
```

可选参数包括 `--wm-mask`、`--fa`、`--template`、`--tracking-config`、`--shell`、`--shell-tolerance`、`--seeds`、`--step-size`、`--max-angle`、`--min-length`、`--max-length`、`--batch-size` 和 `--device`。请运行 `python slicer_backend/run.py --help` 查看当前代码实际提供的完整参数。

运行 `python slicer_backend/run.py --check-dependencies` 可检查后端环境依赖。若只需执行 FA 计算和配准以检查 QC，可加 `--preprocess-only`，此模式不需要模型检查点。官方发布的模型按默认架构配置；若使用按其他架构训练的检查点，需提供匹配的追踪配置（`--tracking-config`），否则模型参数可能无法正确加载。配置文件仅用于模型架构和输入特征参数；本次运行的路径及界面追踪参数仍由命令行或 Slicer 界面提供。

## 验证状态

以下官方 Linux x86_64 版本已在 Xvfb 下通过测试，进程均正常退出。使用相同的插件源码和独立后端环境：

| Slicer | 内置 Python | Qt | VTK | 模块安装与界面测试 | 真实后端合成数据追踪 |
| --- | --- | --- | --- | --- | --- |
| 4.11.20210226 | 3.6.7 | 5.15.1 | 8.2.0 | 通过 | 通过，含 FA-only 配准 |
| 5.2.2 | 3.9.10 | 5.15.2 | 9.1.0 | 通过 | 通过 |
| 5.12.4 | 3.12.10 | 5.15.2 | 9.6.2 | 通过 | 通过，含 FA-only 配准 |

界面测试覆盖 Additional module paths 自动发现、Slicer 实际创建的模块界面、含空格路径、分块 QProcess 结果标记、取消后重启、RAS VTP 加载、ITK 变换求逆和零纤维结果。测试使用独立的设置命名空间，不覆盖用户保存的 DDTracking 输入路径。真实后端测试从界面计算 FA、配准、使用发布检查点追踪，并将非空纤维加载到场景。外部后端实测环境为 Python 3.9.19 / PyTorch 2.4.1；上文的新安装示例选用 Python 3.10。

CPU 合成数据全流程测试使用公开发布的检查点，自动计算 FA、完成刚性配准和追踪，并导出非空原生/MNI 纤维；测试验证纤维长度和两种空间的坐标一致性。可按以下命令重跑（`--model` 指向下载并解压后的发布检查点）：

```bash
python tests/synthetic_pipeline.py --model /path/to/weights.pt
python -m unittest discover -s tests -v
```

对目标 Slicer 运行下面的界面测试；请将两个仓库路径都替换为自己的绝对路径。Windows 使用 `Slicer.exe`；macOS 的可执行文件通常是 `/Applications/Slicer.app/Contents/MacOS/Slicer`。

```bash
/path/to/Slicer --ignore-slicerrc --disable-settings --no-main-window --testing \
  --additional-module-paths /absolute/path/to/DDTracking/slicer/DDTracking \
  --python-script /absolute/path/to/DDTracking/tests/slicer_smoke.py
```

测试需要能从启动 Slicer 时的 PATH 找到独立 Python，也可通过环境变量 `DDTRACKING_TEST_PYTHON` 指定其可执行文件；该轻量测试只运行标准库进程辅助脚本，不需要模型或后端依赖。测试打印 Slicer/Python/Qt/VTK 版本，成功输出 `DDTracking Slicer smoke test passed` 并返回退出码 0，失败返回非零。仅看到通过文字但进程异常退出不算完整通过。

源码安装已经验证；CMake 扩展构建文件已提供，尚未完成使用 Slicer SDK 的正式扩展打包或提交到 Extensions Index。如果自行构建二进制扩展包，需要使用对应 Slicer 版本的 SDK；跨版本使用请优先采用上述源码安装。

在 Linux 无头环境且已安装 Xvfb 时，可运行：

```bash
xvfb-run -a /path/to/Slicer --ignore-slicerrc --disable-settings --no-main-window --testing \
  --additional-module-paths /absolute/path/to/DDTracking/slicer/DDTracking \
  --python-script /absolute/path/to/DDTracking/tests/slicer_smoke.py
```

Windows/macOS 与其他 Slicer 版本可使用同一源码安装步骤，并运行本仓库的界面测试记录该平台/版本的验证结果。兼容目标不等于已逐一测试所有历史和未来版本。

CUDA、Windows 和 macOS 尚未测试；真实受试者数据上的配准/追踪等价性也尚未评估。

## 许可与科研使用

仓库目前没有许可证文件；公开发布不构成源码或模型权重的再分发授权。用于科研时请检查输出并引用主 README 中列出的 DDTracking 论文。
