# 在 3D Slicer 中使用 DDTracking

本仓库包含一个 3D Slicer 模块，可从图形界面启动纤维追踪。Slicer 负责界面和加载原生空间结果；配准、特征准备和模型推理由独立的 Python 环境执行，避免把模型依赖安装进 Slicer 自带的 Python。

## 环境要求

- 已安装 3D Slicer，并能访问 `slicer/DDTracking` 模块目录。Slicer 5.2.2 Linux 已通过 Xvfb 下的界面冒烟测试；其他版本和平台尚未验证。
- 为后端准备独立的 Python 环境。建议使用 Python 3.9 或更高版本，并安装 [`../requirements-slicer-2026-10-22.txt`](../requirements-slicer-2026-10-22.txt) 中的依赖。
- PyTorch 需单独安装。CPU 用户请从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择 CPU wheel；CUDA 用户根据操作系统和 CUDA 环境选择合适的 wheel。所选 PyTorch 必须能在后端 Python 环境中运行；Slicer 本身不需要安装 PyTorch。
- 与数据匹配的 DDTracking 模型检查点。项目在 [GitHub v0.1 release](https://github.com/yishengpoxiao/DDTracking/releases/tag/v0.1) 提供模型权重（压缩包约 154 MB）。下载并解压后，在模块中选择对应的检查点文件。请保留权重的来源信息及其适用条款。

从仓库根目录创建环境的示例：

```bash
conda create -n ddtracking-slicer python=3.10 pip -y
conda activate ddtracking-slicer
python -m pip install -r requirements-slicer-2026-10-22.txt
# 再按上面的官方选择器安装 CPU 或适用于本机的 CUDA 版 PyTorch。
```

依赖文件没有列出 PyTorch，因为正确的安装包取决于运行设备。仅运行 Slicer 模块时不需要安装完整的 `requirements.yaml`；该文件还包含训练和其他用途的包。

## 在 Slicer 中安装模块

1. 克隆或下载本仓库，并将其放在运行 Slicer 期间保持可用的位置。
2. 在 Slicer 中打开 **Edit → Application Settings → Modules**，把仓库中的 `slicer/DDTracking` 目录加入 **Additional module paths**。
3. 重启 Slicer，在模块选择器中打开 **DDTracking**。
4. 在 **External Python** 中选择 `ddtracking-slicer` 环境里的 Python 可执行文件（Linux 通常是环境目录下的 `bin/python`；Windows 通常是 `python.exe`）。
5. 选择必需输入文件和一个已经存在的输出目录。点击 **Register FA only** 可只计算/读取 FA 并执行配准，此模式不需要模型检查点；点击 **Run tracking** 则还需要选择匹配的模型检查点。

模块会异步启动后端，因此运行时 Slicer 仍可响应。日志区域显示后端输出和错误；点击 **Cancel** 可请求终止运行进程。成功完成后，模块会把原生空间纤维加载到当前场景中。结果图像和变换文件保存在输出目录；勾选 QC 选项还可加载配准后的 FA 和 MNI 空间纤维。

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

Slicer 5.2.2 Linux 在 Xvfb 下已通过界面集成测试，覆盖模块界面初始化、分块 QProcess 结果标记读取、取消后重启、RAS VTP 加载、ITK 变换求逆和零纤维结果。已另外验证从插件界面启动真实后端、使用发布检查点完成合成数据追踪，并自动将纤维加载到场景的完整流程。

CPU 合成数据全流程测试使用公开发布的检查点，自动计算 FA、完成刚性配准和追踪，并导出非空原生/MNI 纤维；测试验证纤维长度和两种空间的坐标一致性。可按以下命令重跑（`--model` 指向下载并解压后的发布检查点）：

```bash
python tests/synthetic_pipeline.py --model /path/to/weights.pt
python -m unittest discover -s tests -v
```

Slicer 界面测试可通过 `Slicer --no-main-window --testing --python-script /absolute/path/to/DDTracking/tests/slicer_smoke.py` 运行。源码安装已经验证；CMake 扩展构建文件已提供，尚未完成使用 Slicer SDK 的正式扩展打包或提交到 Extensions Index。

CUDA、Windows 和 macOS 尚未测试；真实受试者数据上的配准/追踪等价性也尚未评估。

## 许可与科研使用

仓库目前没有许可证文件；公开发布不构成源码或模型权重的再分发授权。用于科研时请检查输出并引用主 README 中列出的 DDTracking 论文。
