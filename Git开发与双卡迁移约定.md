# Git 开发与双卡迁移约定

在 `Hyperloop/` 下独立编写面向昇腾 SGLang 的优化项目。AMD Hyperloom 用作设计参考。本地与免费 910B3 共用一条 `main` 开发线，后续迁到双 910B2 完成真机验收。Git 仓库也放在 `Hyperloop/` 下。

| Git 引用 | 用途 |
| --- | --- |
| `main` | 本地和 910B3 共用的开发线 |
| `base-v0.1` | 通用闭环首次跑通、接入 910B3 之前的底稿 |
| `910b3-v0.1` | 910B3 上跑通完整优化闭环的实测提交 |
| `ascend-910b2` | 双 910B2 适配需要修改项目代码时，从已验证版本创建的短期分支 |

## 1. 在本地固定通用底稿

先完成 Coordinator、Session、候选执行、正确性校验和 KEEP/REVERT 裁决，预留环境接入点。在本地用模拟执行器走通闭环，记录标记为 `simulated`。底稿验收后，在 `Hyperloop/` 内初始化 Git，将源码、脱敏配置和项目文档提交到 `main`。模型权重、虚拟环境、凭据和大体积原始记录按各自路径保存，并通过 `.gitignore` 排除。

在底稿提交上打标签。

```bash
git tag -a base-v0.1 -m "接入 910B3 前的通用底稿"
```

## 2. 同步本地与 910B3

用私有远程仓库同步。首次在本地关联 `origin`，推送 `main` 和底稿标签，再从 910B3 克隆。

```bash
git remote add origin "私有仓库地址"
git push -u origin main
git push origin base-v0.1
```

本地和 910B3 都用 `main`，包含同一套昇腾适配代码。可以直接在 910B3 调试，修改后提交并推送，再在本地运行 `git pull --ff-only`。从本地修改时采用相同流程，由 910B3 拉取。测试前用 `git rev-parse HEAD` 核对提交，并用 `git status --short` 检查待提交的源码改动。Python 环境、SGLang 版本、模型路径、设备和 `tp_size` 通过目标配置传入。

910B3 上完成原始基线、Profile、候选校验、独立计时和裁决后，在对应的 Session 记下项目提交号。将 `910b3-v0.1` 打在该实测提交上并推送标签。

```bash
git tag -a -m "910B3 单卡闭环通过" 910b3-v0.1 "实测提交的 SHA"
git push origin 910b3-v0.1
```

## 3. 迁到双 910B2

先比较底稿和 910B3 实测版，梳理环境探测、SGLang 启停、压测、Profile 和候选验证分别做了哪些适配。

```bash
git diff base-v0.1 910b3-v0.1
```

- **只换配置即可运行时**，沿用 `main`，增加双 910B2 的环境和 `tp_size=2` 配置。
- **需要改项目代码时**，从 910B3 已验证提交开短期分支，在该分支完成双卡适配。

```bash
git switch -c ascend-910b2 910b3-v0.1
```

双 910B2 独立开 Session，验证 HCCL，保留原始 `TP=2` 基线、两个 rank 的 Profile、候选正确性结果和关闭采集后的端到端复测。记录实际测试的项目提交和 SGLang 提交。双卡验收后，把通用修正和双卡配置合回 `main`，复跑 910B3 的固定请求，确认同一份代码支持两端。
