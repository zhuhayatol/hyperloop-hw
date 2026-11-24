# Hyperloop

Hyperloop 面向昇腾 SGLang 独立实现 Hyperloom 的优化编排。Coordinator 驱动角色回合、候选准入、任务派发和结果写回。设备执行端负责启动、采集、测量和清理，独立评测层关联每次候选的原始样本和裁决。

当前自带的两套目标和 Agent 响应均为确定性模拟，输出标记为 `simulated`。使用 Python 3.11 或更新版本，可在 Windows 或 Linux 本地执行。真机适配和性能验收分别使用目标设备上的独立 Session。

[通用底稿规划](通用底稿初步规划.md) · [执行架构图](Agent执行架构图.html) · [Git 和双卡迁移](Git开发与双卡迁移约定.md)

## 运行一次完整模拟

从 `Hyperloop/` 目录执行。

```text
python -m pip install -e .
python -m hyperloop run --task configs/task.simulated.json --target configs/target.910b3.simulated.json --reference fixtures/reference.simulated.json --scenario fixtures/scenario.simulated.json --session-root sessions --session-id forge-910b3
```

该样例固定长输入短输出、短输入长输出两组请求，重复计时三次。Orchestration 先直委 Explore，再派 Specialist 提交框架补丁，Critic 按提案 ID 评审。然后阶段机记录算子阶段、成对扫描与回环。第二轮用尽固定响应后收束流程。示例资料给定的吞吐中位数为 100、110、115、122、123 tok/s。最后的 123 比父版 122 高约 0.82%，低于锁定的 2% 门槛。

本地算子阶段演练 GEAK 式服务方案交付和 Forge 式算子补丁交付。设备内核、模型 forward、在线服务三类计时分开归档。Profile 使用模拟采集；上游可选的 Roofline、专项 GEMM 调优与 Fusion 后端留在具体运行适配阶段。

同一任务、参考与 Agent 场景也可搭配双 rank 目标，或在 Session 创建时选择另一条算子路线。下面同时演示这两项变更。

```text
python -m hyperloop run --task configs/task.simulated.json --target configs/target.910b2-tp2.simulated.json --reference fixtures/reference.simulated.json --scenario fixtures/scenario.simulated.json --session-root sessions --session-id geak-910b2 --kernel-backend geak
```

双 rank 目标记录 rank 0 和 rank 1 的模拟 Profile。每次 `run` 使用新的 Session ID。中断后沿用生成的 Session 路径和冻结输入。

## 查看、恢复和独立复测

```text
python -m hyperloop status sessions/simulated/forge-910b3
python -m hyperloop resume sessions/simulated/forge-910b3
python -m hyperloop report sessions/simulated/forge-910b3
python -m hyperloop reproduce sessions/simulated/forge-910b3 R03
```

- `status` 从完整的落盘事件读取阶段、任务状态、已接受版本和候选裁决。
- `run` 和 `resume` 支持 `--max-steps N`。用它在若干调度步骤后暂停，再从同一目录恢复。
- `report` 从相同的事件和证据重建 `summary.json`、`report.md` 和可离线打开的 `report.html`。报告可查看角色与任务交接、验收、Profile 和原始证据引用。
- `reproduce` 使用该候选和父版的冻结规格独立复测。查看返回的复测证据位置和裁决。复测证据单独留存，原 Session 的版本结算保留。

Session 根目录下的 `manifest.json` 保存输入快照，`events.jsonl` 是事件记录，`state.json` 是可重建缓存。报告通过证据索引指向各次运行的原始文件。模拟数据可核对流程与口径，真机收益按真实环境独立建立基线。

查看 `rounds/<候选>/evidence_bundle.json` 可定位父版、原始基线、参考输出和指标数据。每次派发的完整回执位于 `evidence/actions/`，独立 Critic 的裁决依据位于 `evidence/decisions/`。 `profile/` 保存各 rank 采集索引，`sweep/` 保存并发点对照与跳过原因。

## 在新 Session 参考旧配方

完成的 Session 若写出了 `evidence/recipes/final.json`，可在新一轮运行时通过 `--recipe` 导入。

```text
python -m hyperloop run --task configs/task.simulated.json --target configs/target.910b3.simulated.json --reference fixtures/reference.simulated.json --scenario fixtures/scenario.simulated.json --session-root sessions --session-id forge-910b3-recipe --recipe sessions/simulated/forge-910b3/evidence/recipes/final.json
```

导入时核对来源会话已关闭、配方是该会话登记的最终产物、证据摘要对应原始文件。新 Session 在预检后核对模型、tokenizer、工作负载、目标设备与实际环境。匹配的参数作为 Agent 回合参考，仍由新 Session 建立原始基线、提交候选并独立校验和计时。两套卡型与 TP 配置分别验收。

## 本地验收

```text
python -m unittest discover -s tests -v
```

验收覆盖问题输入、权限和直接委派、候选预审、租约释放、独立证据门、两条算子路径、成对扫描和回环恢复。启用故障路径的测试可使用同一场景中的 `base-unavailable` 初始版本，复验后锚定首个能运行的 `base-fixed`。
