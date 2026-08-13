# Calibration Test Fixtures

## `synthetic_fit_results_with_p.csv`

这是从真实 `k_preprocess.py` → `k_estimate.py` 链路产出（`RAW_fit_results.csv`）中抽取的代表性子集。

- **用途**：为 `k_estimate.py` 提供确定性测试输入，验证回归与可视化功能
- **非原始数据**：这是阶段一 `k_preprocess` 的中间产物，不是原始相机捕获数据
- **结构**：
  - 40 行数据
  - 4 个亮度 bin（`MeanMin`/`MeanMax` 组合）
  - 2 种极性（`P=0` 和 `P=1`）
  - 每个 bin 含多个 `DiffMin`/`DiffMax` 区间（对应不同 `kdL` 值）
- **列名**：`P, MeanMin, MeanMax, DiffMin, DiffMax, MuHat, LambdaHat, Mu, Sigma, Count`

这些列与 `k_estimate.py` 的 `--input` CSV 输入要求一致。`Lbar` 和 `kdL` 是 `k_estimate.py` 内部
从 `MeanMin`/`MeanMax` 和 `DiffMin`/`DiffMax` 计算得到的，不是输入列。
