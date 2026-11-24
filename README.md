# Hyperloop

Hyperloop 面向昇腾 SGLang 独立实现 Hyperloom 的优化编排。Coordinator 驱动角色回合、候选准入、任务派发和结果写回。设备执行端负责启动、采集、测量和清理，独立评测层关联每次候选的原始样本和裁决。

当前自带的两套目标和 Agent 响应均为确定性模拟，输出标记为 `simulated`。使用 Python 3.11 或更新版本，可在 Windows 或 Linux 本地执行。真机适配和性能验收分别使用目标设备上的独立 Session。

[从优化任务到交付](DOC/优化任务与交付架构.html) · [执行架构图](DOC/Agent执行架构图.html) · [运行与复核](DOC/运行与复核.md) · [通用底稿规划](DOC/通用底稿初步规划.md) · [开发日志](DOC/开发计划与修改日志.md) · [本地验收记录](DOC/本地验收记录.md) · [Git 和双卡迁移](DOC/Git开发与双卡迁移约定.md)
