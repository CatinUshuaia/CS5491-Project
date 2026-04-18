# CVRP 启发式搜索与 LLM 项目详细介绍

## 1. 项目概述

本项目围绕容量车辆路径问题（CVRP）构建了一套可复现的实验框架，目标不是直接训练一个端到端模型，而是让系统在一组可解释的启发式表达式中进行搜索，并比较以下几类方法在不同数据场景下的表现：

- 传统 baseline 求解器，如 `nearest_neighbor`、`greedy`、`ortools`
- 基于表达式的启发式搜索
- 使用 mock 扰动生成候选表达式的搜索流程
- 使用 LLM 生成候选表达式的搜索流程

项目当前已经从最初的 notebook 原型演进为可批量执行的正式工程版，支持：

- classic 标准 CVRP 数据处理与 benchmark
- fresh 生鲜场景数据扩展与 benchmark
- 正式 benchmark、消融实验、mock vs LLM 对照实验
- 表格、图像与结果文件自动导出

---

## 2. 项目目标与核心思路

项目的核心思想是把“如何选择下一个客户节点”表示为一个可执行的 Python 算术表达式。例如：

```python
dist_matrix[current][c] - instance['demands'][c]
```

给定这样的表达式，系统会把它转换成一个打分函数，在构造 CVRP 路径时为每个候选客户打分，分数越小越优先。围绕这一思想，项目形成了完整搜索闭环：

1. 准备一组 seed expression 作为初始候选。
2. 在一批实例上评估每个表达式的表现。
3. 按可行率、目标值、路线数、复杂度等指标排序。
4. 保留 top 表达式。
5. 用 mock 或 LLM 生成下一轮候选。
6. 重复上述过程，形成 outer loop。

这种设计的优点是：

- 可解释：最终得到的是可读的启发式表达式，而不是黑盒参数。
- 可控：可以明确约束候选复杂度、变量范围与安全性。
- 可扩展：可以方便接入新数据集、新目标函数、新搜索策略。

---

## 3. 当前项目结构

项目主要目录如下：

- `01_raw_data/`：原始 CVRPLib 数据及快照
- `02_processed_data/`：处理后的 classic / fresh 数据
- `03_core_algorithm/`：核心算法模块、方法实现与 notebook 原型
- `04_experiment_outputs/`：benchmark、对照实验输出结果
- `05_scripts/`：正式 pipeline 脚本入口
- `06_docs/`：数据说明、流程文档、交接说明

当前应优先把 `05_scripts/` 与 `03_core_algorithm/modules/` 视为正式主链路，notebook 更多用于原型开发、展示和历史追溯。

---

## 4. 数据体系

### 4.1 classic 数据

classic 数据来自 CVRPLib。脚本 `05_scripts/process_cvrplib.py` 会完成：

- 解析 `.vrp` 与 `.sol`
- 构造标准化 `base.json`
- 生成 `meta.json`
- 导出 `index.csv`
- 生成 classic QA 报告与数据 schema 文档

处理后的 classic 数据位于：

- `02_processed_data/classic/base/`
- `02_processed_data/classic/meta/`
- `02_processed_data/classic/index.csv`

### 4.2 fresh 数据

fresh 数据不是外部独立数据集，而是在 classic 基础上扩展出的生鲜配送场景版本。脚本 `05_scripts/generate_fresh_dataset.py` 会为每个 classic 实例补充：

- 时间窗
- 服务时间
- 新鲜度等级
- 最大可接受运输时长
- 温区
- 迟到惩罚
- 损耗惩罚
- fresh 场景目标权重

处理后的 fresh 数据位于：

- `02_processed_data/fresh/fresh/`
- `02_processed_data/fresh/fresh_meta/`

这使得项目同时支持两类问题：

- classic：以距离为主目标的标准 CVRP
- fresh：同时考虑距离、迟到、损耗惩罚的扩展 CVRP

---

## 5. 内部数据表示与关键约定

无论 classic 还是 fresh，项目都会先把实例标准化为统一的内部 `instance` 格式，再交给 solver 和搜索流程使用。

关键约定如下：

### 5.1 Depot 统一为 0

内部统一使用 `depot = 0`。这意味着：

- route 表示以 `0` 作为仓点
- `distance_matrix`、`demands`、`coords` 都要与这个编号体系对齐

### 5.2 classic 关键字段

classic `instance` 至少包含：

- `name`
- `depot`
- `demands`
- `capacity`
- `num_nodes`
- `distance_matrix`
- `raw`

### 5.3 fresh 关键字段

fresh 在 classic 基础上增加：

- `service_time_min`
- `ready_time_min`
- `due_time_min`
- `freshness_class`
- `max_travel_time_min`
- `late_penalty_per_min`
- `spoilage_penalty`
- `objective_weights`

### 5.4 表达式接口

classic 表达式常见变量包括：

- `dist_matrix[current][c]`
- `dist_matrix[c][instance['depot']]`
- `instance['demands'][c]`
- `remaining`

fresh 表达式常见变量包括：

- `travel_to_c`
- `est_lateness`
- `est_spoil`
- `instance['demands'][c]`
- `remaining`

表达式最终都必须输出一个标量分数，且默认规则为：**分数越小，优先级越高**。

---

## 6. 核心算法模块

### 6.1 baseline 求解器

项目包含三类 baseline：

- `nearest_neighbor`
- `greedy`
- `ortools`

其中：

- `nearest_neighbor` 和 `greedy` 提供轻量级可解释基线
- `ortools` 提供强基线参考

在 fresh 场景中，虽然 `ortools` 路径距离通常更短，但由于它没有直接围绕 fresh 的综合目标做定制，因此在迟到/损耗惩罚计入后不一定最优。

### 6.2 表达式驱动启发式

项目的关键桥梁是“把表达式变成 score function”，再用这个 score function 驱动构造式求解器：

- classic：基于距离、需求、剩余容量等变量排序客户
- fresh：在 classic 基础上增加对迟到和损耗的动态估计

因此，搜索流程的本质是在搜索“更好的客户选择规则”。

### 6.3 候选评估与聚合

项目会对每个候选表达式在多实例上做批量评估，并汇总为表达式级结果。

classic 常见聚合指标：

- `feasible_rate`
- `avg_cost`
- `avg_runtime_sec`
- `avg_num_routes`

fresh 常见聚合指标：

- `feasible_rate`
- `avg_objective`
- `avg_distance`
- `avg_late_penalty`
- `avg_spoil_penalty`
- `avg_num_routes`
- `avg_runtime_sec`

### 6.4 搜索增强模块

为提升搜索效率和候选质量，项目实现了三类增强机制：

#### 重复过滤

不仅做字符串级去重，还做：

- 语法规范化去重
- 基于 probe 实例的轻量行为去重

#### 复杂度控制

使用 AST 结构复杂度近似衡量表达式复杂度，用于：

- 过滤过于复杂的候选
- 在排序时偏向更简洁表达式

#### novelty 支持

用聚合行为签名而不是表达式文本来跟踪“是否真正带来新行为”，其目的在于：

- 减少重复探索
- 鼓励候选多样性

### 6.5 排序逻辑

当前排序逻辑遵循：

1. 优先更高的可行率
2. 再比较更低的目标值
3. 再比较更少的路线数
4. 再比较更低的复杂度

同时会引入一个 `mo_score` 作为多目标辅助 tie-break，保证排序更稳定。

---

## 7. LLM 与 mock 候选生成

项目当前支持两种候选生成方式。

### 7.1 mock 生成

mock 方式本质上是对 top 表达式做局部扰动，快速生成一批结构相近的变体。它的特点是：

- 便宜
- 稳定
- 适合做 baseline 搜索生成器

### 7.2 LLM 生成

LLM 方式会根据 top 表达式构造 prompt，请模型返回 JSON 形式的新表达式列表。随后系统会做：

- JSON 解析
- 安全性检查
- 复杂度过滤
- 去重

当前脚本支持通过环境变量配置：

- `CVRP_OPENAI_API_KEY`
- `CVRP_OPENAI_HOST`

如果本地没有配置 API key，对照脚本会自动退化为 mock-only。

---

## 8. 正式实验入口

当前推荐执行顺序如下：

1. `python 05_scripts/process_cvrplib.py`
2. `python 05_scripts/generate_fresh_dataset.py`
3. `python 05_scripts/run_formal_benchmark.py`
4. `python 05_scripts/run_llm_vs_mock_small.py`
5. `python 05_scripts/run_fresh_formal_benchmark.py`
6. `python 05_scripts/run_fresh_llm_vs_mock_small.py`

这 6 个脚本覆盖了当前项目的完整主流程。

---

## 9. 项目进度与验证状态

截至目前，项目已完成以下工作：

- classic 数据标准化处理
- fresh 生鲜数据扩展生成
- classic / fresh 核心算法模块脚本化
- 正式 benchmark 脚本封装
- 消融实验自动化执行
- mock vs LLM 对照实验
- 结果导表与图像导出
- 完整全流程正式运行验证

---

## 10. 实验结果摘要

### 10.1 运行规模

本轮正式全流程执行情况如下：

- classic / fresh 均加载 `95` 个实例
- classic formal benchmark：`29` 个实例，`4` 个 seed
- fresh formal benchmark：`29` 个实例，`3` 个 seed
- classic mock vs LLM：`29` 个实例，`3` 个 mock seed + `3` 个 LLM seed
- fresh mock vs LLM：`29` 个实例，`3` 个 mock seed + `3` 个 LLM seed
- 全部流程总耗时约 `72.34` 分钟

### 10.2 classic 结果

- formal benchmark 最优 `best_avg_cost_mean = 1148.54`
- 所有消融配置最终最优值一致，说明当前搜索流程在 benchmark 上已收敛到同一水平
- baseline 中 `ortools` 最优，`avg_cost = 946.79`
- 结论：classic 搜索优于 `nearest_neighbor` 和 `greedy`，但仍未追平 `ortools`

### 10.3 classic mock vs LLM

- mock 最优聚合结果：`1260.07`
- LLM 最优聚合结果：`1280.89`
- LLM 相对 mock 退化约 `1.65%`

说明：

- 当前 classic 表达式空间中，LLM 尚未表现出稳定优于 mock 的生成优势
- LLM 能生成更多变体，但并未突破当前 mock 已能找到的较优规则

### 10.4 fresh 结果

- formal benchmark 最优 `best_avg_cost_mean = 355783.50`
- 所有消融配置最终最优值一致
- baseline 中最优为 `nearest_neighbor`
- 搜索最优结果与 `nearest_neighbor` 持平

说明：

- 当前 fresh 搜索流程还没有找到比基础 `travel_to_c` 更好的规则
- fresh 目标中损耗惩罚占比很高，是当前总 objective 的主要来源

### 10.5 fresh mock vs LLM

- mock 与 LLM 聚合结果完全持平
- 两者全部 seed 的最佳表达式都回到了 `travel_to_c`

说明：

- 当前 fresh 表达式空间和 prompt 设计还没有把生鲜场景特征真正转化为更优策略
- 这并不表示 LLM 无用，而是说明当前实验设置下它的边际收益尚未被释放

---


## 11. 主要结果产物位置

### 正式 benchmark

- `04_experiment_outputs/formal_benchmark/`
- `04_experiment_outputs/fresh_formal_benchmark/`

重点文件：

- `ablation_seed_summary.csv`
- `ablation_aggregate_summary.csv`
- `formal_experiment_meta.json`

### mock vs LLM

- `04_experiment_outputs/llm_vs_mock_small/`
- `04_experiment_outputs/fresh_llm_vs_mock_small/`

重点文件：

- `baseline_summary.csv`
- `mock_vs_llm_seed_summary.csv`
- `mock_vs_llm_aggregate_summary.csv`
- `llm_token_summary.json`

### 数据与 QA 文档

- `06_docs/pipeline_docs/data_schema.md`
- `06_docs/pipeline_docs/fresh_data_schema.md`
- `06_docs/pipeline_docs/qa_report.md`
- `06_docs/pipeline_docs/fresh_qa_report.md`

---

## 12. 维护与接手时的重要注意事项

### 12.1 不建议轻易改动的部分

以下约定被多处默认依赖：

- `depot = 0`
- 统一 `instance` schema
- 表达式只返回单标量 score
- 排序优先保证可行性

如果调整这些约定，需要同时检查：

- solver
- evaluation
- 候选过滤
- LLM prompt
- outer loop
- 导出结果逻辑

### 12.2 notebook 与正式代码的关系

当前仓库中的 advanced notebook 仍然有参考价值，但它们更适合：

- 回看算法原型
- 理解函数来源
- 做快速交互式试验

真正用于正式执行时，应优先使用：

- `03_core_algorithm/modules/`
- `05_scripts/`

### 12.3 如何理解当前结果

目前结果不应解读为“LLM 不适合做该任务”，而应理解为：

- 当前表达式空间较窄
- 当前 prompt 还不够强
- 当前搜索机制尚未充分放大 LLM 的长处

因此，LLM 当前没有显著优于 mock，更像是在说明：

**方法设计仍是瓶颈，样本量已经不是主要瓶颈。**

---

## 13. 后续优化建议

如果后续还要继续推进，建议按以下优先级进行。

### 优先级 1：改进 fresh 方法设计

重点包括：

- 扩大 fresh 表达式变量集合
- 让候选显式建模 `lateness` / `spoilage` / `distance` 的 trade-off
- 重新设计 LLM prompt
- 调整候选过滤与排序策略

### 优先级 2：在方法改进后再补一轮 benchmark

当前 benchmark 已足够支持阶段性结论；下一轮更大规模实验更适合放在方法改进之后，而不是立刻继续盲目扩量。

### 优先级 3：补充更面向报告的分析

例如：

- classic 相对 `known_opt_cost` 的 gap 分析
- fresh 目标分解分析
- 不同规模实例下的表现差异
- 表达式复杂度与效果的相关性

---

## 14. 总结

本项目已经从最初的 notebook 原型发展为一套较完整的 CVRP 启发式搜索实验工程，具备以下特点：

- 同时支持 classic 与 fresh 两类场景
- 支持 baseline、表达式搜索、mock、LLM 对照
- 支持正式 benchmark 与消融实验
- 支持结果导出、图表生成与文档沉淀

当前版本最大的价值在于：

- 流程稳定
- 结果可复现
- 结构清晰
- 便于接手和继续演进

当前版本最大的局限在于：

- classic 尚未超过 `ortools`
- fresh 尚未找到超越 `travel_to_c` 的更优规则
- LLM 相比 mock 还没有形成稳定优势


