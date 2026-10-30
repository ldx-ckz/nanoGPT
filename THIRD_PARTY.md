# 参考来源

## 代码与教程

- 作者：Andrej Karpathy。
- 项目：[karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)。
- 教程：[Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)。
- 本仓库的 `GPT2.py` 是学习复现中的实现，包含跟随教程编写及参考上游的代码，初始化时从本地已有文件原样复制。

本仓库是个人学习复现项目。引用、修改和分发参考代码时，应遵守对应来源的适用许可并保留相关声明。本次初始化未给参考代码或整个项目额外指定新的许可证。

## 文本样例

- 根目录 `input.txt` 是原有项目中的 Shakespeare 文本样例。
- 它与本地保存的上游 `input.txt` 的 SHA-256 相同：
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`。
- 上游文件：[build-nanogpt/input.txt](https://github.com/karpathy/build-nanogpt/blob/master/input.txt)。

此样例用于跑通训练流程，不是 FineWeb-Edu 数据集。
