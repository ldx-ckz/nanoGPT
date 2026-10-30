# nanoGPT2

跟随 [Andrej Karpathy 的 build-nanogpt](https://github.com/karpathy/build-nanogpt) 教程，从零复现 GPT-2，并记录模型实现、训练优化和实验过程。


## 当前进度

以下表示代码已经包含对应实现，不代表所有功能都已完成独立验证。

- [x] GPT-2 模型结构、因果注意力、权重共享和初始化。
- [x] `input.txt` 分词和批次加载。
- [x] AdamW、梯度裁剪、预热与余弦学习率。
- [x] BF16、TF32、SDPA、`torch.compile` 和梯度累积。
- [x] DDP 多卡启动与梯度同步逻辑。
- [ ] 修复并验证数据加载器的读取位置推进和多卡边界。
- [ ] 接入 FineWeb-Edu 分片和独立验证集。
- [ ] 保存检查点、恢复训练和生成样例。
- [ ] 记录可复现的训练指标和模型质量评估。

当前默认参数为 12 层、12 个注意力头、768 隐藏维度、1024 上下文；训练词表填充至 50304。`B=8`，总 batch 为 524288 tokens，训练 50 步。

## 仓库结构

```text
nanoGPT2/
├── GPT2.py              # 当前模型与训练入口
├── input.txt            # 当前训练入口实际读取的小型文本样例
├── requirements.txt     # 训练和预训练权重加载的直接依赖
├── requirements-local.txt # 本地 Windows CUDA 环境的版本清单
├── THIRD_PARTY.md       # 参考代码及样例数据来源
├── data/README.md       # 数据约定与后续准备方式
├── scripts/README.md    # 后续辅助脚本的放置约定
└── results/README.md    # 实验记录格式；尚无实测结果
```

## 环境与运行

### 本地 Windows 环境

当前本地配置为 Python 3.12.14、PyTorch 2.14.0+cu130、tiktoken 0.14.0、transformers 4.57.1；对应 NVIDIA RTX 4060 Laptop GPU。`requirements-local.txt` 记录本地包及其依赖的具体版本。创建环境时使用已安装的 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

在 VS Code 的“Python: Select Interpreter”中选择 `.venv\Scripts\python.exe`。本地环境可用于编辑、导入及 GPU 小实验；下述完整训练入口仍面向 Linux，Windows 下的 NCCL 多卡和原生 `torch.compile` 训练未验证。

### 云端 Linux 环境

当前训练入口面向 Linux + NVIDIA CUDA GPU，脚本包含直接调用 CUDA 和 NCCL 的逻辑，CPU 或 macOS 训练尚未适配。

先进入仓库根目录，检查环境：

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

若尚未安装 PyTorch，按照 [官方安装页面](https://pytorch.org/get-started/locally/) 选择与 GPU 和驱动匹配的 CUDA 构建。

`requirements.txt` 包含训练和加载预训练权重所需的直接依赖，其中 PyTorch 保持不锁定，以接受云镜像已有的 CUDA 构建。云端不要直接套用本地的 CUDA 13.0 配置；实际云端环境版本尚未记录。

单卡启动：

```bash
python -u GPT2.py
```

同一台机器双卡启动：

```bash
torchrun --standalone --nproc_per_node=2 GPT2.py
```

运行前先处理下文列出的已知问题。两种启动方式均为使用说明，本仓库初始化时未执行 GPU 训练。


