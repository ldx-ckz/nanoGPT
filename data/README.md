# 数据约定

当前 `GPT2.py` 使用根目录的 `input.txt`，这里尚未接入训练入口。

后续数据目录：

- `raw/`：原始下载数据，不提交到 Git。
- `processed/`：分词结果和分片，不提交到 Git。

加入新数据时，记录来源、配置、预处理命令、tokenizer、训练/验证拆分和数据规模，并在代码中同步更新读取路径。

FineWeb-Edu 作为后续任务；本仓库尚未包含完整数据下载脚本。可参考上游 [fineweb.py](https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py)，接入前确认下载空间、缓存目录和持久化位置。
