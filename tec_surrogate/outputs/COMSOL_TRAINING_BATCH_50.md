# COMSOL 新增训练批次

- CSV：`comsol_training_batch_50.csv`
- 样本编号：`100-149`
- 优先批次：`100-139`
- 追加批次：`140-149`（计算预算允许时）
- 固定工况：报告中的 COP 网格保持现有设置；`dT0_K` 列仅用于记录，当前批跑工具不会填写它。

在批跑工具中选择该 CSV。先运行 `100-139`，确认报告归档正常后，再运行 `140-149`。

`expected_n_pairs`、`expected_n_length`、`expected_n_width` 和
`expected_leg_height_mm` 是预检列，批跑工具会忽略，不需要手工删除。
