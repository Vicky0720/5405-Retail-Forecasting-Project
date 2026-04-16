# Week 1 Forecasting Pipeline

## 2026-04-16 当前提交状态

本轮已补齐前置评估项：`course_prophet_pipeline.py` 现在使用 6 个 rolling origins，其中前 4 个作为 `selection`，最后 2 个作为 `holdout`。模型权重、residual correction、bias correction 都只在 `selection` 上拟合或生成候选，然后统一进入 leaderboard，由 `holdout` 选择最终提交模型。

当前自动选择结果为 `item_prophet`。这并不表示 residual/bias correction 被忽略，而是说明它们已经入榜评估，但在 holdout 上没有超过 `item_prophet`。这一步的意义是避免把 attribution 中发现的历史误差直接外推，造成看似修正、实际泛化更差的结果。

本轮关键输出包括：

- `outputs/course_prophet_pipeline_backtest_20220915.csv`：按 origin、split、model 汇总的 rolling backtest。
- `outputs/course_prophet_model_leaderboard_by_split_20220915.csv`：按 `selection/holdout` 拆分的模型榜单。
- `outputs/course_prophet_bias_corrections_20220915.json`：由 attribution 训练出的 bias correction 系数。
- `outputs/course_prophet_selected_model_20220915.json`：holdout 自动选择出的最终模型。
- `outputs/forecast_submission_20220915_prophet_pipeline_selected.json`：已提交到 API 的最终预测文件。

当前目录中的 `forecast_week1.py` 已经升级为一个“多模型、多轮回测、分 bucket 集成”的 week 1 预测流程。

最新的课程 pipeline 版本在 `course_prophet_pipeline.py` 中：它按 Session7 思路组合 `cate2/item` top-down、Prophet、bottom-up mix 和 weighted `1-MAPE` 权重搜索，用来生成更少机械重复感的提交结果。

当前线上提交采用 Prophet pipeline 后的 residual-corrected 版本：在保持 Prophet 非重复趋势/季节结构的基础上，加入近期 aggregate residual pattern，以减少过度平滑。

如果你更喜欢交互式使用，也可以直接打开：

- `forecast_week1_workflow.ipynb`

## 当前能力

- 连接课程 API 或读取离线 CSV
- 生成 `option_id × day` 粒度未来 28 天预测
- 自动输出比赛提交 JSON
- 自动做滚动回测，并区分 `selection` 与 `holdout`
- 输出按整体、按 horizon、按 bucket 的评估结果
- 输出整体最优权重与按 bucket 最优权重

## 当前模型结构

脚本同时生成 4 路候选预测：

1. `wbaseline`
   - 统计基线。
   - 使用最近 `7/28/56` 天水平、星期季节性、截断趋势与稀疏收缩。

2. `etr`
   - `ExtraTrees` 的 direct multi-horizon 全局模型。
   - 每个 horizon 单独训练。
   - 按 `high_volume / mid / sparse` 分群建模。

3. `lgbm`
   - `LightGBM` 的 direct multi-horizon 全局模型。
   - 每个 horizon 单独训练。
   - 同样按 `high_volume / mid / sparse` 分群建模。

4. `item_topdown`
   - 先在 `item` 层用 `baseline + ETS` 做预测。
   - 再按近期 option share 拆分回 `option`。

## 主要特征

当前特征已经不只是 lag 和 rolling，额外引入了业务侧信息：

- 日期特征：`dow / dom / month / weekofyear / is_weekend / trend_idx`
- 时序特征：`lag_1/7/14/21/28/35/42/56`
- 滚动特征：`roll_mean_* / roll_zero_*`
- item 层级特征：`item_lag_* / item_roll_mean_* / option_share_in_item`
- 价格特征：`unit_sales_price / unit_label_price / discount_ratio / discount_depth / price_gap_to_label`
- 相对价格特征：`relative_price_in_item / price_rank_in_item`
- 上下线特征：`online_state / online_state_change / days_since_online_begin`
- 促销强度特征：`item_promo_intensity`
- 分群特征：`vol56 / nz56 / segment`
- 事件特征：`event_name / event_type`

## 回测设计

当前默认设置：

- 总回测 origin 数：`10`
- holdout origin 数：`2`
- horizon bucket：`1-7 / 8-14 / 15-21 / 22-28`

流程如下：

1. 用较早的 `selection` origins 比较候选模型。
2. 在 `selection` 上搜索整体最优权重。
3. 在 `selection` 上继续搜索各个 horizon bucket 的最优权重。
4. 用最后 2 个 `holdout` origins 检查泛化表现。
5. 用 bucket 权重生成最终正式预测。

## 当前结果

基于 `2022-07-14` 这次运行：

- 整体最优权重：`etr = 1.0`
- bucket 权重：
  - `1-7`: `etr 0.7 + item_topdown 0.3`
  - `8-14`: `etr 1.0`
  - `15-21`: `etr 0.9 + lgbm 0.1`
  - `22-28`: `etr 1.0`

聚合回测结果：

- `ALL_SELECTION`
  - `ensemble_selected = 0.443991`
  - `ensemble_bucket_selected = 0.444655`
- `ALL_HOLDOUT`
  - `ensemble_selected = 0.554941`
  - `ensemble_bucket_selected = 0.555141`

说明：

- 从整体上看，当前最强单模型是 `etr`
- 从分 bucket 看，短期与中长期仍有少量组合收益
- bucket 集成在 holdout 上略优于整体单一权重集成

## 输出文件

运行完成后会生成：

- `history_option_day_<feature_date>.csv`
- `forecast_option_day_<feature_date>.csv`
- `forecast_submission_<feature_date>.json`
- `backtest_scores_<feature_date>.csv`
- `backtest_predictions_long_<feature_date>.csv`
- `backtest_metrics_by_horizon_<feature_date>.csv`
- `backtest_metrics_by_bucket_<feature_date>.csv`
- `selected_weights_<feature_date>.json`
- `selected_weights_by_bucket_<feature_date>.json`

## 运行方式

### API 模式

```powershell
python forecast_week1.py --mode api --access-token "<YOUR_TOKEN>" --output-dir .\outputs
```

### 离线 CSV 模式

```powershell
python forecast_week1.py --mode csv --history-csv .\sample_history.csv --output-dir .\outputs
```

离线 CSV 至少应包含：

- `option_id`
- `date`
- 一个销量字段，例如 `sales_quantity`

## 测试

```powershell
python -m unittest test_forecast_week1.py
```

## 下一步可继续优化

1. 把 horizon 粒度的最优模型选择进一步做成自动 gating，而不只是 bucket 权重。
2. 继续扩展更稳的价格弹性与促销记忆特征。
3. 对高销量 option 引入更细的参数搜索，对稀疏 option 增加更强的保守约束。
4. 引入更长周期的稳定性检验，避免只对最近几个 origin 调得过细。
