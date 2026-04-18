# CVRP Project

基于 CVRPLib 的容量车辆路径问题（CVRP）实验项目，包含：

- 原始数据标准化处理（classic）
- 面向生鲜配送场景的扩展数据生成（fresh）
- 启发式/基线方法对比与自动化 benchmark
- mock 与 LLM 候选表达式搜索对照实验
- 实验结果导出与图表生成

## 当前目录结构

- `01_raw_data/`：原始 CVRPLib 数据
- `02_processed_data/`：标准化后的 classic/fresh 数据
- `03_core_algorithm/`：核心算法模块与 notebook
- `04_experiment_outputs/`：实验输出
- `05_scripts/`：可直接运行的 pipeline 脚本入口
- `06_docs/`：流程、数据结构、交接文档

## 环境要求

- Python 3.10+
- 推荐使用虚拟环境（venv/conda）
- 主要依赖：
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `ortools`
  - `openai`（仅在 LLM 对照实验中需要）

## 快速开始

1) 创建并激活虚拟环境（可选）  
2) 安装依赖  
3) 按 pipeline 顺序执行脚本

示例（Windows PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy pandas matplotlib ortools openai
```

## 推荐执行顺序

1. 处理 CVRPLib 原始数据（classic）

```powershell
python 05_scripts/process_cvrplib.py
```

2. 生成 fresh 扩展数据

```powershell
python 05_scripts/generate_fresh_dataset.py
```

3. 运行正式 benchmark

```powershell
python 05_scripts/run_formal_benchmark.py
```

4. 运行 classic mock vs LLM 对照 benchmark

```powershell
python 05_scripts/run_llm_vs_mock_small.py
```

5. 运行 fresh 正式 benchmark

```powershell
python 05_scripts/run_fresh_formal_benchmark.py
```

6. 运行 fresh mock vs LLM 对照 benchmark

```powershell
python 05_scripts/run_fresh_llm_vs_mock_small.py
```

## LLM 配置说明

`run_llm_vs_mock_small.py` 与 `run_fresh_llm_vs_mock_small.py` 会优先读取环境变量：

- `CVRP_OPENAI_API_KEY`
- `CVRP_OPENAI_HOST`（可选，默认使用脚本内配置）

建议在本地设置环境变量，不要把真实密钥写入仓库。

## 主要输出位置

- `01_raw_data/cvrplib_sets/`：classic 原始 `.vrp/.sol` 数据
- `02_processed_data/classic/`：classic 标准数据与索引
- `02_processed_data/fresh/`：fresh 数据与元信息
- `04_experiment_outputs/`：benchmark 和对照实验输出

## 建议阅读入口

- `README.md`：项目总体说明与运行入口
- `06_docs/handover_notes/CVRP_LLM_Handover_Notes.md`：项目结构、方法和接手说明


## 许可证

本项目采用 MIT License。详见 `LICENSE`。

