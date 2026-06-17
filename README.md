# 装备损耗趋势拆解与预测

本仓库用于完成数理统计大作业：基于时间序列分解与高斯过程回归，对俄乌冲突中俄罗斯装备损耗的每日序列进行趋势拆解、模型比较和未来 60 天预测。

实际建模数据来自公开 GitHub 镜像(active)：

`PetroIvaniuk/2022-Ukraine-Russia-War-Dataset`

使用文件为：

`russia_losses_equipment.json`

该 JSON 记录俄罗斯各类装备的累计损耗。脚本会将累计值差分为每日新增损耗，再进行分解和预测。

## 代码文件

核心代码位于：

`src/analyze_equipment_losses.py`

主要输出：

- `outputs/model_metrics.csv`：模型指标表。
- `outputs/forecast_60d.csv`：未来 60 天逐日预测。
- `outputs/analysis_summary.json`：样本范围、负增量数量、累计值等摘要。
- `outputs/figures/01_series_trend_shocks.svg`：每日损耗、趋势线和突发点。
- `outputs/figures/02_decomposition.svg`：趋势、周季节项和残差分解。
- `outputs/figures/03_forecast.svg`：未来 60 天预测和不确定性带。

## 主流程

`main()` 是脚本入口，完整流程如下：

1. 调用 `ensure_dirs()` 创建输出目录。
2. 调用 `load_equipment()` 读取装备损耗 JSON，并完成口径统一、差分、负增量修正和元数据统计。
3. 取广义装备每日损耗 `broad_equipment` 作为主序列。
4. 对主序列做 `log(1+y)` 变换，以缓解高波动和异方差。
5. 调用 `decompose_weekly()` 进行 STL 类分解，得到趋势项、周季节项、残差项和突发点。
6. 分别调用 `fit_ets()`、`fit_arima_family(..., seasonal=False)`、`fit_arima_family(..., seasonal=True)` 拟合经典时间序列模型。
7. 调用 `fit_gp()` 拟合三种高斯过程核函数：RBF、Periodic、Matern 3/2。
8. 调用 `write_outputs()` 生成图表、模型指标、预测 CSV、摘要 JSON 。

## 函数说明

### `ensure_dirs()`

创建输出目录 `outputs/` 和图表目录 `outputs/figures/`。如果目录已存在，不会报错。

### `load_equipment()`

读取 `data/russia_losses_equipment.json`，并完成数据预处理。

主要步骤：

- 将 JSON 转为 `pandas.DataFrame`。
- 将 `date` 字段转换为日期类型并排序。
- 处理车辆与油罐字段口径变化：早期使用 `military auto + fuel tank`，后期使用 `vehicles and fuel tanks`。
- 对累计装备损耗做一阶差分，得到每日新增损耗。
- 将负增量截断为 0。
- 构造 `broad_equipment`、`heavy_equipment` 和 `drone_only` 三个序列。
- 返回每日数据表和摘要元数据。

### `moving_average(y, window)`

计算居中移动平均，用于估计长期趋势。

在本项目中，`decompose_weekly()` 使用 31 日窗口平滑每日损耗序列，以降低单日峰值对趋势项的影响。

### `decompose_weekly(dates, y)`

执行 STL 类分解。

输出包括：

- `trend`：31 日移动平均趋势项。
- `seasonal`：按星期几估计的 7 日季节项。
- `remainder`：原序列减去趋势项和季节项后的残差。
- `shock`：是否为突发点的布尔数组。
- `shock_threshold`：突发识别阈值。

突发点使用残差的 `Q3 + 1.5 IQR` 规则识别。

### `FitResult`

模型结果数据类，用于统一保存不同模型的输出。

字段包括：

- `name`：模型名称。
- `params`：模型参数。
- `fitted`：样本内拟合值。
- `forecast`：未来预测均值。
- `lower`：预测区间下界。
- `upper`：预测区间上界。
- `aic`：AIC。
- `bic`：BIC。
- `waic`：WAIC，主要用于高斯过程模型。

### `gaussian_ic(resid, k)`

基于高斯残差近似计算 AIC、BIC 和残差方差。

公式为：

```text
AIC = -2 log L + 2k
BIC = -2 log L + k log n
```

其中 `k` 是参数数量，`n` 是残差样本量。

### `fit_ets(y_log, horizon)`

拟合 Holt 线性趋势形式的 `ETS(A,A,N)`。

实现方式：

- 在 `log(1+y)` 空间建模。
- 对 `alpha` 和 `beta` 做网格搜索。
- 使用 AIC 选择最佳参数。
- 输出未来 `horizon` 天预测及近似 95% 区间。

该模型用于提供可解释的指数平滑基准。

### `make_lag_matrix(series, lags)`

根据给定滞后阶数构造滞后回归矩阵。

例如 `lags=[1,2,7]` 时，每一行包含截距项、前 1 日差分、前 2 日差分和前 7 日差分。该函数服务于 ARIMA 和 SARIMA 近似模型。

### `recursive_forecast_diff(diff, lags, coef, horizon)`

使用已估计的滞后回归系数递归预测未来差分值。

每预测一天，就把预测值追加到历史序列中，再用于下一天预测。这样可以生成多步预测路径。

### `fit_arima_family(y_log, horizon, seasonal)`

拟合 ARIMA 或 SARIMA 近似模型。

当 `seasonal=False` 时：

- 模型近似为 `ARIMA(p,1,0)`。
- 在一阶差分序列上搜索短期滞后阶数。

当 `seasonal=True` 时：

- 模型近似为带周季节滞后的 SARIMA。
- 在短期滞后基础上加入 7 日和 14 日滞后。

函数使用最小二乘估计滞后系数，并用 AIC 选择最佳滞后结构。

### `kernel_matrix(x1, x2, kind, length, period, amp)`

计算高斯过程核矩阵。

支持三种核：

- `RBF`：平方指数核，表示平滑变化。
- `Periodic`：周期核，表示固定周期结构。
- `Matern32`：Matern 3/2 核，表示局部粗糙变化。

该函数是高斯过程模型的核心。

### `gp_log_marginal(y, k)`

计算高斯过程的对数边际似然，并返回 Cholesky 分解和求解后的 `alpha`。

实现细节：

- 使用 Cholesky 分解提高数值稳定性。
- 若核矩阵不可分解，则返回负无穷似然，表示该参数组合不可用。

### `fit_gp(y_log, horizon)`

拟合三种高斯过程模型。

主要流程：

- 只使用最近 500 天训练，以减少不同战争阶段混合造成的偏差。
- 对 RBF、Periodic、Matern32 三种核分别搜索长度尺度、幅度和噪声参数。
- 用对数边际似然选择每种核的最佳参数。
- 计算未来 60 天预测均值和预测方差。
- 调用 `approximate_gp_waic()` 计算近似 WAIC。

本文结果中，`GP-Matern32` 的 WAIC 最低，因此作为主要解释模型。

### `approximate_gp_waic(y, mu, sigma)`

用模拟方式近似计算 WAIC。

实现方式：

- 固定随机种子 `20260602`。
- 围绕预测均值生成 300 组正态模拟。
- 计算逐点 log likelihood。
- 根据 `lppd` 和有效参数惩罚项计算 WAIC。

该函数主要用于比较不同 GP 核函数，而不是与 AIC 直接跨类别比较。

### `line_svg(...)`

生成简单 SVG 折线图。

支持：

- 多条折线。
- 日期轴标签。
- 突发点散点标记。
- 预测区间阴影带。

脚本用它生成 3 张核心图表，不依赖 `matplotlib`。

### `write_outputs(df, meta, decomp, models, horizon)`

统一写出所有分析结果。

主要输出：

- 趋势和突发点图。
- 分解图。
- 预测图。
- 模型指标 CSV。
- 未来 60 天预测 CSV。
- 数据摘要 JSON。

该函数还会根据 WAIC 选择最佳 GP，根据 AIC 选择最佳经典模型。

### `fmt_num(x)`

格式化数字，将数值转为带千分位逗号的字符串。例如 `520168` 会显示为 `520,168`。

### `main()`

脚本入口函数，串联完整分析流程。用户通常只需要运行该脚本，无需手动调用各个函数。

## 模型解释提醒

本项目中的模型结果应按以下方式理解：

- 原始公开数据是“报告损耗”，不应直接等同于独立核实的真实战场损耗。
- STL 是“STL 类分解”或“参考 STL 思想的近似分解”，不是标准软件包 STL。
- ARIMA 和 SARIMA 是基于差分滞后回归的近似实现。
- AIC/BIC 主要用于经典模型内部比较。
- WAIC 主要用于三个 GP 核函数内部比较。
- 预测区间比单一点预测更重要，因为冲突型序列存在强不确定性。

## 主要结果

核心数据摘要：

- 样本范围：2022-02-25 至 2026-06-02。
- 样本量：1559 天。
- 广义装备累计值：520,168。
- 重装备累计值：190,169。
- 负增量数量：13 个。
- 最大负修正：-83。

核心模型结果：

- 经典模型中，ARIMA 的 AIC 最低，为 1590.6。
- 高斯过程中，GP-Matern32 的 WAIC 最低，为 1994.5。
- ARIMA 未来 60 天日均预测约为 2349.4。
- GP-Matern32 未来 60 天日均预测约为 1879.6。

结论上，ARIMA 适合作为经典基准，GP-Matern32 更适合解释局部粗糙变化和预测不确定性。
