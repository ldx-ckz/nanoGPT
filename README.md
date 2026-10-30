# nanoGPT2

跟随 [Andrej Karpathy 的 build-nanogpt](https://github.com/karpathy/build-nanogpt) 教程，从零复现 GPT-2，并记录模型实现、训练优化和实验过程。

当前以 `GPT2.py` 为主线，保留单文件结构；之后通过实际 Git 提交记录每一步变化。此版本来自本地已有代码，不自动包含云服务器上尚未同步的修改。

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
├── requirements.txt     # 训练入口的基础依赖
├── THIRD_PARTY.md       # 参考代码及样例数据来源
├── data/README.md       # 数据约定与后续准备方式
├── scripts/README.md    # 后续辅助脚本的放置约定
├── notes/progress.md    # 当前进度、已知问题和后续任务
└── results/README.md    # 实验记录格式；尚无实测结果
```

虚拟环境、完整数据、检查点和大量日志由 `.gitignore` 排除。`checkpoints/` 是运行时目录，空目录不会被 Git 跟踪。

## 环境与运行

当前训练入口面向 Linux + NVIDIA CUDA GPU，建议先使用云平台已经装好 CUDA 版 PyTorch 的 Python 3.11 环境。脚本包含直接调用 CUDA 和 NCCL 的逻辑，CPU 或 macOS 训练尚未适配。

先进入仓库根目录，检查环境：

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

若尚未安装 PyTorch，按照 [官方安装页面](https://pytorch.org/get-started/locally/) 选择与 GPU 和驱动匹配的 CUDA 构建。基础依赖文件未锁定版本；实际实验时应记录所用版本。

单卡启动：

```bash
python -u GPT2.py
```

同一台机器双卡启动：

```bash
torchrun --standalone --nproc_per_node=2 GPT2.py
```

运行前先处理下文列出的已知问题。两种启动方式均为使用说明，本仓库初始化时未执行 GPU 训练。

代码会读取 `GPT2.py` 同目录下的 `input.txt`，首次使用 `tiktoken` 可能联网下载分词编码资源。显存不足时可在代码中减小 `B`，同时保持总 batch 的整除约束。第一次运行 `torch.compile` 会有编译开销。

如果要调用 `GPT.from_pretrained()`，另行安装可选依赖：

```bash
python -m pip install transformers
```

## 当前限制

为保留已有实现的起点，初始化仓库时没有修改 `GPT2.py` 的训练行为：

1. `DataLoaderLite.next_batch()` 用 `process_rank` 推进位置，单卡 rank 为 0 时会反复读取同一批数据；多卡的推进和边界也需要一起修复和验证。
2. 训练结束没有保存模型、优化器和数据读取状态，不能直接恢复训练。
3. 生成代码在 `sys.exit(0)` 之后，不会执行；其中的分词器变量也需要在生成路径中初始化。
4. 没有独立验证集或完整 FineWeb 训练结果，不能据此声明已经完成 GPT-2 质量复现。

详细任务见 [进度记录](notes/progress.md)。

## 如何记录复现过程

- 每完成一个明确变化就提交一次，例如修复加载器、加入验证集、加入检查点。
- 原理理解和问题分析写在 `notes/`；代表性指标、图表和生成样例写在 `results/`。
- 用 Git 历史保存代码演进；不需要复制出 `GPT2_v1.py`、`GPT2_final.py` 等版本。
- 从云端同步代码时，先下载最新文件到本仓库，检查 `git diff`，再提交。
- 检查点保存在磁盘并单独备份；Git 提交不包含模型权重。

## GitHub

这是本地独立 Git 仓库，尚未绑定 GitHub 远程地址。创建 GitHub 上名为 `nanoGPT2` 的空仓库后，用页面显示的真实地址配置：

```bash
git remote add origin <GitHub 仓库地址>
git push -u origin main
```

上面的尖括号内容需要替换，不能原样执行。参考来源和许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
