# 本次修复怎么检查、怎么运行

请使用当前仓库中的代码，不要混用旧副本。下面的路径是示例，
运行时替换成你克隆仓库的实际位置。最近一次验证的范围见 `VALIDATION.md`。

## 1. 先看什么

- `CORRECTIONS.md`：逐项说明发现的问题、修复方法、算法依据和影响范围。
- `verification_outputs/corrected_v2/RESULTS.md`：实际验证结果。
- `verification_outputs/corrected_v2/deadline_reference_check.csv`：负 gap 的数值解释。
- `verification_outputs/corrected_v2/targeted/`：deadline、maintenance、random 的新数据和曲线。
- `verification_outputs/corrected_v2/library_smoke/policy_run_status.csv`：11 个主实例的快速运行检查。

## 2. Jupyter 运行

先重启 kernel，避免内存里仍然保留旧版模块。然后运行：

```python
%cd "C:/path/to/rmab-benchmark-suite"
%run "smoke_test.py"
```

完整重做本次验证：

```python
%run "verify_corrections.py"
```

其中针对性实验使用 3 个实例、8 个策略、N=20/100/500、每个组合 3 个 seed，
每次统计 200 步。这是修复验证，不是完整论文实验。Q-Whittle 没有在这里重新训练。
再次运行会复用本次验证的新缓存，原来的旧缓存仍然保留。
验证脚本还会自动再读取一次这些缓存，逐个核对 seed 对应的奖励是否完全相同。

只做更小的检查可以运行：

```python
%run "verify_corrections.py" --quick
```

快速检查单独存到 `verification_outputs/corrected_v2_quick/`，不会覆盖完整验证结果。

## 3. 单独理解负 gap

```python
from benchmark_api import run_named_experiment

result = run_named_experiment(
    "deadline_S10_a20", "Myopic",
    N=500, horizon=200, seed=123,
    include_finite_horizon_bound=True,
)
print("200 步平均奖励:", result["mean_reward"])
print("稳态 LP 参考值:", result["lp_upper_bound"])
print("相对稳态差值:", result["relative_gap"])
print("200 步 LP 参考值:", result["finite_horizon_lp_bound"])
print("相对有限时间差值:", result["finite_horizon_relative_gap"])
```

该确定性例子中，前两个数是 0.091175 和 0.09，有限时间 LP 也是 0.091175。
相对稳态差值为负，有限时间差值在浮点精度范围内为零。

加上 `burn_in=20` 表示先运行 20 步，再统计后面 200 步。此时实际共运行 220 步。
返回的轨迹数组保留全部 220 步，`mean_reward` 只统计后 200 步。
可选的有限时间 LP 始终比较全部 220 步，不能把它和后 200 步的均值混用。
20 步对这里的 Myopic 已足够，不代表对其他实例和算法也足够。

## 4. 缓存怎么切换

普通运行会自动读取**修复后的**兼容缓存。单次强制重算：

```python
result = run_named_experiment(
    "deadline_S10_a20", "FTVA_Strategy",
    N=100, horizon=200, seed=123,
    force_recompute=True,
)
```

下次把 `force_recompute` 删除或设为 `False`，就恢复读取新缓存。
不能让新代码直接复用修复前的轨迹，否则取整错误会重新进入结果。
旧缓存没有删除，可以留作历史核对；正常的新实验会自动避开它们。

## 5. 完整论文矩阵

```python
%run "run_instance_matrix_benchmark.py"
```

仍使用原来的 11 个实例、9 个策略、5 个 N 和 20 个 seed，默认统计 200 步，
默认 burn-in 为 0。这个命令比较慢，包含 Q-Whittle 的训练。
新输出在 `instance_matrix_outputs/corrected_v2/`，不会覆盖原来的历史 CSV 和图。
负 gap 会如实显示。FTVA 能运行不等于已经证明其在每个例子上都渐近最优。

完整实验尚未重做之前，不要把小规模验证结果替代论文中的完整结论。
论文汇总入口现在默认读取 `instance_matrix_outputs/corrected_v2/`，不再回退到旧目录。
缺少数据或版本不一致会明确报错。完整矩阵跑完后再运行汇总即可。

## 6. 这次验证的结论与边界

2026-09-19 的检查通过了 21 项回归测试、88 次全实例快速模拟，以及 216 次针对性模拟。
后续报告入口检查将测试扩充到 34 项；加入 Q-Whittle 后，针对性矩阵扩展为 243 次运行。
9 月 20 日补查计时和异质入口后，回归测试共 44 项，均已通过。
FTVA 在 deadline 和 maintenance 上原来的除零失败已经解决。
默认配置仍是 200 步，不是长期极限；这里没有通过调参数来保证某个算法胜出。

这次没有替换 LP-Priority 的排序方法或 LP-Update 的优化目标。
不过两者调用的公共模拟器已经修正，因此不能再说这两个老师文件“完全没修改”。
`ATTRIBUTION.md` 已说明修改归属。

尚未重做：完整 20-seed 论文矩阵及完整计算成本实验。
这些旧结果不能直接标成已修复版本。GitHub 也尚未推送这次本地修改。

## 7. 修正后的报告汇总

先预览已经验证过的小规模数据，不需要再运行模拟：

```python
%run "generate_benchmark_findings.py" --known-model-dir verification_outputs/corrected_v2/targeted --output-dir verification_outputs/corrected_v2/report_preview
%run "generate_paper_figures.py" --known-model-dir verification_outputs/corrected_v2/targeted --output-dir verification_outputs/corrected_v2/report_preview
%run "check_paper_readiness.py" --known-model-dir verification_outputs/corrected_v2/targeted --output-dir verification_outputs/corrected_v2/report_preview
```

这个预览会注明 `preliminary subset`，不会装成完整论文实验。
`report_inputs.json` 记录输入目录、数据指纹、代码指纹和失败情况。
热力图保留负 gap；排名只使用所有选定算法都有完整结果的共同实例。
`policy_comparison.csv` 是描述性的奖励排名，不再把 reward、gap、beta 加权凑成“综合冠军”。
排名差异也不代表统计显著性。

完整矩阵计算完成后，运行上面三个脚本时不加参数即可，结果写入
`paper_summary_outputs/corrected_v2/`。旧计时、异质和 learning 数据不会自动混入。
旧的 `generate_paper_draft.py` 是历史模板，默认阻止生成，避免旧结论重新进入论文；
本次没有直接修改你的正式 LaTeX 文稿。

## 8. 加上 Q-Whittle 的验证

```python
%run "verify_corrections.py" --with-qwhittle
```

这个版本只对三个重点实例增加 Q-Whittle，仍保持每个 penalty 50,000 步、
41 个 penalty、gamma=0.95、训练 seed=2026。Monte Carlo seed 改变的是评估轨迹，
不是重新训练模型，因此还不能据此评价训练随机性的影响。

结果独立存入 `verification_outputs/corrected_v2_with_qwhittle/`。
同一次运行现在也会自动生成该目录下的 `report_preview/`，不需要另跑汇总命令。
需要预览这份九算法结果时，将第 7 节命令中的数据目录改为
`verification_outputs/corrected_v2_with_qwhittle/targeted`，输出目录改为
`verification_outputs/corrected_v2_with_qwhittle/report_preview`。

## 9. 计时和异质部分的补查

在重启后的 Jupyter 内核中运行：

```python
%cd "C:/path/to/rmab-benchmark-suite"
%run "verify_extensions.py"
```

这条命令重新计算，不读取或删除奖励缓存。输出位于
`verification_outputs/corrected_extensions_v2/`：

- `timing/`：三个实例、八个策略、N=20/100、T=20，每个设置两个 seed；48 次冷启动测量、96 次模拟，无失败。没有缩短训练后冒充 Q-Whittle 冷启动；本轮明确不测它。
- `five_types/`：maintenance 和 wireless，每个实例五套 P、R，N=10/20、T=10，七个策略、两个 seed；56 次模拟，无失败。
- `unique_models/`：相同流程，但 N=10 时有 10 套模型、N=20 时有 20 套；56 次模拟，无失败。
- `RESULTS.md`、`verification_summary.json`：协议和结果汇总。每个子目录均有 CSV、曲线图和代码指纹。

本轮修正了这些具体问题：

1. 异质 LP-Update 原来会跨 seed 保留状态求解结果，后一次计时可能用了前一次的缓存。现在只在一次模拟内复用，下一次模拟开始就清空，不改变优化目标。
2. 所有策略初始化失败时，两个实验入口现在仍会保存每次失败，不再因空表而崩溃丢掉记录。
3. 计时策略使用实际预算 `floor(alpha*N)/N`；不同预算比例的 setup 和时间拟合分开统计。
4. 异质模拟先检查动作是不是 0/1，不再把小数或负数动作直接转成整数。
5. 新结果默认写入 `corrected_v2/` 子目录，历史结果不覆盖。总计时收集入口检查代码指纹，不自动混入旧 CSV。

`simulation_seconds` 包含初始化、决策、奖励和环境转移，不是纯粹的决策时间。
异质 setup 目前每个配置只测一次；这批运行时间短、N 只有两个取值，不能据此宣称复杂度或综合最优策略。
完整计时、异质和全部实例的 20-seed 论文矩阵仍需另行运行。

## 10. Unknown-model / online learning 补查

```python
%cd "C:/path/to/rmab-benchmark-suite"
%run "verify_online_learning.py"
```

独立输出目录是 `verification_outputs/corrected_online_v2/`。
这个小实验包含 random、maintenance、wireless、deadline 四个实例，
N=20/50，每组两个 seed，每次从头学习 200 步，共 112 次模拟。
正式的大实验入口仍是 `run_unknown_model_benchmark.py`，但默认写入
`unknown_model_outputs/corrected_v2/`，不覆盖历史结果。

### 算法分类不要混淆

- `KnownWhittleOracle`、`KnownLPPriorityOracle`：知道真实 P、R 的参考曲线，不是 unknown-model。
- `OnlinePlugInWhittle`：根据观测估计 P、R，再计算指数，属于 model-based online learning。
- `OnlineQLearningIndex`：不估计 P，按传统折扣 Q-learning 更新，用 Q(s,1)-Q(s,0) 排序。它是一个启发式基线，不是学习 Whittle subsidy 的 Q-Whittle。
- `OnlineRewardGreedy`、`OnlineUCBReward`：估计即时奖励的简单基线。
- `OnlineRandom`：随机选择固定数量的 arm，不学习模型。

这部分做了检查和修正：

1. 用一个禁止读取 P、R 的测试对象验证了所有未知模型策略的构造与样本更新路径；没有发现它们偷读真模型。
2. 所有 arm，包括 passive arm，都会提供状态、奖励和转移观测；奖励为 R[s,a]，没有额外奖励噪声。N 个同质 arm 共用一张学习表，每步得到 N 条样本。
3. 初始化时间 `setup_seconds` 与模拟时间 `simulation_seconds` 分开；总时间 `runtime_seconds` 是两者之和。后者含 reset、交互和学习，不是纯决策时间。
4. Plug-in Whittle 求指数失败时仍允许回退到即时奖励优势，但新增 `num_index_failures`、`num_fallback_decisions`、`last_index_error` 记录，不能再把这种运行当成全程使用 Whittle 指数。
5. 动作必须是真正的 0/1，预算使用经过浮点整数校正的 floor(alpha*N)，oracle 的 LP 也使用这个实际比例。
6. 失败和不完整 seed 组保留在 CSV；不完整组不会作为完整曲线展示。实验参数和源代码指纹写入 `experiment_settings.json`。

`mean_reward` 是全程每 arm 平均奖励；`tail_mean_reward` 是最后 25% 步的平均奖励，
本次对应最后 50 步。learning curve 在选定时间点对 seed 求平均，没有进行窗口平滑，
也不是 cumulative reward 或 regret 曲线。学习过程中 N 增大也会增加可用样本数，
不要直接把这些曲线当成已知模型策略的纯 N 收敛实验。

### 本轮实测结果

2026-09-20：全套 58 项回归测试通过，日志在
`verification_outputs/corrected_online_v2/unit_tests.txt`。
本轮 112 次在线模拟全部完成，56 个参数组都有完整的两个 seed。
Plug-in Whittle 共进行了 80 次指数重算，没有求解失败，也没有使用回退决策。
详细协议与计数见同目录的 `RESULTS.md` 和 `verification_summary.json`。

代码来源与记录检查通过不代表短期学习已经收敛；论文推荐算法仍需完整实验。
历史缓存和实验结果未删除或覆盖；它们不能代替修正后的实验结果。

## 11. Deadline 扩展验证

```python
%run "verify_deadline.py" --long
```

输出写入 `verification_outputs/release_v2/`。九个策略、五个 N、20 个 seed、
每次 200 步，共 900 次运行均成功；另有 FTVA 的 15 次 4,000 步运行均成功。
这解决了原来的执行失败，不代表 FTVA 在该例子上最优，也不等于验证了全部理论条件。
2026-09-27 重新检查了全部 58 项回归测试，并验证上述 915 次缓存结果可以完整复用。
