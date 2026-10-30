# 实验结果

尚未录入实测训练结果。后续每次有值得保留的实验，创建一份 Markdown 记录；少量指标、曲线和生成样例可以一起提交。

建议使用下面的格式：

```markdown
# 实验名称

- 日期：
- 代码提交 SHA（运行时版本）：
- GPU 型号 / 数量 / 显存：
- Python / PyTorch / PyTorch CUDA / 驱动版本：
- 数据来源 / tokenizer / token 数量 / 训练验证拆分：
- 模型配置：
- B / T / 总 batch tokens / 梯度累积次数：
- 学习率 / 预热步数 / 总训练步数：
- 精度 / torch.compile：
- 启动命令：
- 训练 loss / 验证 loss：
- 预热后稳定吞吐（tokens/s）/ 耗时：
- 生成样例：
- 检查点位置及外部备份位置：
- 结论和限制：
```

完整日志放在忽略的 `logs/` 或 `runs/` 中；模型权重放在忽略的 `checkpoints/` 中。此处只保留能够解释实验结果的必要材料。
