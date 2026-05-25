# Seismic Flow 5D

本仓库用于基于 Flow Matching 和 DiT/Transformer 主干网络的地震 5D 插值重建。程序以不规则采样地震道作为观测上下文，通过 query-context 方式预测规则网格上的缺失道，并将预测结果回填到 SEG-Y 文件中。

## 一、整体工作流

完整处理链路如下：

```text
SEG-Y 原始数据
  |
  | 1. 道头解析与 H5 转换
  |    tool/convert_tool/Segy2H5.py
  |    tool/convert_tool/batch_segy2h5.py
  v
H5 数据
  - irregular H5: 不规则观测道
  - mask H5: 规则网格上的缺失/已有状态
  - label H5: 规则网格标签或完整目标
  |
  | 2. 规则网格匹配与 patch/query-context 预计算
  |    tool/reg_tool/core.py
  |    tool/reg_tool/precompute_anchor_patch_v2.py
  |    tool/reg_tool/auto_params.py
  v
NPZ 索引文件
  - train_pool_idx_2d.npz: 训练用观测池
  - infer_query_context.npz: 推理用 query/context 索引
  - coord_norm_stats.npz: 坐标归一化统计
  |
  | 3. 训练
  |    run_train.sh -> train.py
  v
模型权重与日志
  - resultsFPM/.../checkpoints/model-*.pth
  - resultsFPM/.../logs/training_config.json
  |
  | 4. 推理与 SEG-Y 回填
  |    run_infer.sh -> infer_cli.py -> infer.py
  v
输出结果
  - filled_missing.sgy
  - residual.sgy
  - summary.json
  - filled/unfilled key CSV
```

推荐执行顺序：

1. 选择 SEG-Y 道头配置：设置 `SEGY_CONFIG=field1031`、`sw06` 或 `segc3`。
2. 将 SEG-Y 转换为 H5：生成 `*_irregular.h5`、`*_mask.h5`、`*_label.h5`。
3. 预计算训练和推理索引：生成 `train_pool_idx_2d.npz` 和 `infer_query_context.npz`。
4. 训练 Flow Matching 模型：生成 checkpoint 和训练配置。
5. 推理并回填 SEG-Y：读取 checkpoint、H5、mask SEG-Y 和推理索引，输出补全后的 SEG-Y。

## 二、运行环境

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

| 类别 | 依赖 |
| --- | --- |
| 深度学习 | `torch>=2.0`, `accelerate>=0.20`, `torchdiffeq>=0.2`, `timm>=0.9`, `einops>=0.7` |
| 数据处理 | `numpy>=1.21`, `h5py>=3.0`, `segyio>=1.9`, `pyyaml>=6.0` |
| 日志与可视化 | `tensorboard>=2.10`, `matplotlib>=3.5`, `tqdm>=4.60` |

注意：`run_train.sh` 在 `NUM_GPUS>1` 时会调用仓库根目录的 `accelerate_config.yaml`，但当前仓库未包含该文件。若未自行补充该配置，请使用：

```bash
NUM_GPUS=1 bash run_train.sh
```

## 三、代码目录说明

```text
.
├── run_train.sh                    训练 shell 入口
├── train.py                        训练 Python 入口
├── run_infer.sh                    推理 shell 入口
├── infer_cli.py                    推理、SEG-Y 回填、报告生成入口
├── infer.py                        query-context 推理循环
├── fpm.py                          FlowMatchingModel 封装
├── requirements.txt                Python 依赖
│
├── config/
│   ├── data_config.py              数据集和 query-context 参数解析
│   ├── segy_config.py              SEG-Y preset 加载器
│   └── segy_config.yaml            不同数据集的道头字节位置和排序规则
│
├── dataset/
│   ├── __init__.py                 对外导出 DatasetH5_all_queryctx、DatasetH5Interp
│   ├── dataset_reg.py              主 query-context 数据集
│   └── dataset_interp.py           单 H5 规则网格插值数据集
│
├── model/
│   ├── seisdit_trace_axis.py       SeisDiTRopeV2 与相关模块
│   └── rope.py                     SegmentedRoPEExpCached 坐标位置编码
│
├── transport/
│   ├── transport.py                Transport、Sampler、loss 和采样封装
│   ├── path.py                     Linear、GVP、VP path
│   └── integrators.py              ODE/SDE 积分器
│
├── utils/
│   ├── coord_utils.py              坐标归一化、物理 RoPE 频率推断
│   ├── sampler_utils.py            diverse_topk 等采样工具
│   └── segy_utils.py               SEG-Y 读写、key 查找、排序
│
└── tool/
    ├── convert_tool/
    │   ├── Segy2H5.py              单/三文件 SEG-Y 转 H5
    │   ├── batch_segy2h5.py        多 SEG-Y 并行转 H5
    │   ├── dataset_config.py       batch 转换配置
    │   └── run_convert.sh          固定示例路径的转换 wrapper
    │
    └── reg_tool/
        ├── core.py                 binning、anchor_patch、csg、crg、kdtree 主逻辑
        ├── precompute_anchor_patch_v2.py  生产版 anchor patch 预计算入口
        ├── auto_params.py          根据观测系统估计预计算超参数
        ├── patch_sampler.py        patch、query/context 构造
        ├── anchor_selector.py      FPS、facility location、value-based 锚点选择
        ├── run_core.sh             core.py shell wrapper
        └── run_precompute.sh       precompute_anchor_patch_v2.py shell wrapper
```

## 四、SEG-Y 配置工作流

SEG-Y 道头位置集中在 `config/segy_config.yaml`，由 `config/segy_config.py` 加载。训练、推理、转换和数据集模块都会使用这些配置。

### 4.1 选择 preset

推荐使用环境变量：

```bash
SEGY_CONFIG=field1031 bash run_train.sh
SEGY_CONFIG=field1031 bash run_infer.sh
```

也可以在 Python 中显式加载：

```python
from config import segy_config
segy_config.load_config("field1031")
```

### 4.2 当前 preset

| preset | 用途 | 道头 key 规则 |
| --- | --- | --- |
| `field1031` | 默认数据格式 | 从道头读取 `shot_line`, `shot_stake`, `recv_line`, `recv_stake` |
| `sw06` | 另一套道头字节位置 | 从道头读取同一组 key，但字节位置不同 |
| `segc3` | 坐标自计算模式 | 只读取 `shot_x`, `shot_y`, `rec_x`, `rec_y`，line/stake 由缩放坐标计算 |

默认复合 key 为：

```text
(shot_line, shot_stake, recv_line, recv_stake)
```

该 key 同时用于 H5 对齐、推理预测结果匹配和 SEG-Y 回填。

## 五、工作流 1: SEG-Y 转 H5

### 5.1 三文件转换: `tool/convert_tool/Segy2H5.py`

适用场景：已有三份 SEG-Y 文件，分别表示不规则观测、mask 模板和规则标签。

命令示例：

```bash
python tool/convert_tool/Segy2H5.py \
  --irr /path/to/irregular.sgy \
  --mask /path/to/mask.sgy \
  --label /path/to/label.sgy \
  --dataset-name field1031 \
  --mode fixed \
  --config field1031
```

输出目录由三份 SEG-Y 的公共父目录决定，程序会自动创建 `h5/` 子目录：

```text
<common_root>/h5/field1031_irregular.h5
<common_root>/h5/field1031_mask.h5
<common_root>/h5/field1031_label.h5
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--irr` | 无 | 不规则观测 SEG-Y。与 `--mask`、`--label` 同时提供时启用三文件模式 |
| `--mask` | 无 | mask 模板 SEG-Y，通常缺失道为零值 |
| `--label` | 无 | 规则网格标签或完整数据 SEG-Y |
| `--dataset-name` | `dataset` | 输出 H5 文件名前缀 |
| `--mode` | `self_computed` | `fixed` 表示从道头读取 line/stake；`self_computed` 表示由坐标缩放计算 line/stake |
| `--group-name` | `1551` | H5 内部 group 名 |
| `--compute-ovt` | 关闭 | 是否写入 OVT 字段，如 `mx`, `my`, `hx`, `hy`, `fold` |
| `--config` | 无 | SEG-Y preset 名称，例如 `field1031`、`sw06`、`segc3` |

H5 主要字段：

| 字段 | 说明 |
| --- | --- |
| `data` | 地震道矩阵，形状为 `[N_trace, N_sample]` |
| `sx`, `sy`, `rx`, `ry` | 炮点和检波点坐标 |
| `delta`, `t0` | 采样间隔和起始时间 |
| `shot_line`, `shot_no`, `recv_line`, `recv_no` | 炮线/炮号/检波线/检波号 |
| `shot_stake`, `recv_stake`, `cmp`, `cmp_line`, `offset` | `fixed` 模式下写入的附加道头字段 |
| `sx_original`, `sy_original`, `rx_original`, `ry_original` | `self_computed` 模式下写入的原始坐标 |
| `trace_idx` | 原 SEG-Y 道序 |

### 5.2 示例 wrapper: `tool/convert_tool/run_convert.sh`

`run_convert.sh` 是一个固定路径示例，会调用 `Segy2H5.py`：

```bash
SEGY_CONFIG=field1031 bash tool/convert_tool/run_convert.sh
```

注意：该脚本内的 `--irr`、`--label`、`--mask` 路径是硬编码示例路径。换数据时建议直接使用 `Segy2H5.py` 命令，或先修改脚本内路径。

### 5.3 多文件批量转换: `tool/convert_tool/batch_segy2h5.py`

适用场景：多个 SEG-Y group 需要写入同一个 H5 文件。输入列表由 `tool/convert_tool/dataset_config.py` 的 `segyPairs` 和 `info_h5` 控制。

命令示例：

```bash
python tool/convert_tool/batch_segy2h5.py --num-workers 4 --gzip-level 1
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-workers` | `4` | 并行转换进程数 |
| `--compute-ovt` | 关闭 | 是否计算并写入 OVT 字段 |
| `--mx-bin`, `--my-bin`, `--hx-bin`, `--hy-bin` | `None` | OVT 分箱大小，空值时自动估计 |
| `--keep-temp` | 关闭 | 合并完成后是否保留临时 H5 |
| `--gzip-level` | `1` | H5 gzip 压缩等级，1 最快，9 压缩率最高 |
| `--chunk-ntrace` | `128` | `data` 数据集 trace 维 chunk 大小 |
| `--chunk-nsample` | `256` | `data` 数据集 time sample 维 chunk 大小 |

`dataset_config.py` 中 `segyPairs` 的典型结构：

```python
info_h5 = "/path/to/output.h5"
segyPairs = {
    "1551": ["/path/to/input.sgy", "/path/to/label.sgy", "interp", "5d_line_by_order", "none"],
}
```

## 六、工作流 2: 预计算 patch 与 query/context 索引

预计算的目标是把 H5 中的观测道和规则网格道组织成训练/推理可直接读取的 `.npz` 索引文件。

推荐主线：

```bash
python tool/reg_tool/auto_params.py \
  --raw-h5 /path/to/field1031_irregular.h5 \
  --regular-h5 /path/to/field1031_label.h5 \
  --output-format both

python tool/reg_tool/core.py anchor_patch \
  --raw_h5 /path/to/field1031_irregular.h5 \
  --regular_h5 /path/to/field1031_label.h5 \
  --target_h5 /path/to/field1031_mask.h5 \
  --patch-dir /path/to/anchor_patch_v2 \
  --enable-auto-params \
  --skip-infer
```

如果需要同时生成推理索引，请不要传 `--skip-infer`：

```bash
python tool/reg_tool/core.py anchor_patch \
  --raw_h5 /path/to/field1031_irregular.h5 \
  --regular_h5 /path/to/field1031_label.h5 \
  --target_h5 /path/to/field1031_mask.h5 \
  --patch-dir /path/to/anchor_patch_v2 \
  --enable-auto-params
```

### 6.1 自动估参: `tool/reg_tool/auto_params.py`

该脚本读取 raw H5 和 regular H5 的坐标范围、规则网格维度、覆盖率和 mask 情况，估计预计算参数。

命令示例：

```bash
python tool/reg_tool/auto_params.py \
  --raw-h5 /path/to/field1031_irregular.h5 \
  --regular-h5 /path/to/field1031_label.h5 \
  --group-key 1551 \
  --target-block-volume 400 \
  --output-format both
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-dir` | 内置示例路径 | 同时推断 raw/regular 默认路径 |
| `--raw-h5` | `<base>/raw5d_data1104.h5` | 不规则观测 H5 |
| `--regular-h5` | `<base>/reg5dbin_label1031.h5` | 规则网格 H5 |
| `--group-key` | `1551` | H5 group |
| `--output-format` | `shell` | `shell`、`json` 或 `both` |
| `--target-block-volume` | `400` | 推理 4D block 目标格点数 |

输出参数包括：

| 输出项 | 含义 |
| --- | --- |
| `NUM_ANCHORS`, `ANCHOR_STRIDE` | 训练锚点数量和步长 |
| `K_PATCH`, `TOP_L`, `NUM_QUERY` | patch 宽度、候选邻域、训练 query 数 |
| `BLOCK_DIVISORS`, `STRIDE_DIVISORS` | 推理 4D block 和 stride 的分割系数 |
| `MAX_QUERY_PER_PATCH` | 单个推理 patch 中最多 query 道数 |
| `METRIC_WEIGHTS` | `(sx, sy, rx, ry)` 加权距离权重 |

### 6.2 生产版预计算: `tool/reg_tool/precompute_anchor_patch_v2.py`

该脚本是 `anchor_patch_debug.ipynb` 的生产版 CLI。它会读取 raw/regular H5，归一化坐标，构建 4D grid map，生成训练池和推理 query/context 文件。

命令示例：

```bash
python tool/reg_tool/precompute_anchor_patch_v2.py \
  --raw-h5 /path/to/field1031_irregular.h5 \
  --regular-h5 /path/to/field1031_label.h5 \
  --target-h5 /path/to/field1031_mask.h5 \
  --patch-dir /path/to/anchor_patch_v2 \
  --group-key 1551 \
  --num-anchors 7896 \
  --k-patch 256 \
  --top-l 512 \
  --num-query 8 \
  --beta 0.3
```

路径参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-dir` | `tool/h5/dongfang` 的解析路径 | 默认数据目录 |
| `--raw-h5` | `<base>/raw5d_data1104.h5` | 不规则观测 H5 |
| `--regular-h5` | `<base>/reg5dbin_label1031.h5` | 规则网格 H5 |
| `--target-h5` | 无 | mask/target H5，用于补充读取 mask |
| `--group-key` | `1551` | H5 group |
| `--patch-dir` | `<base>/patchV2` | 输出索引目录 |
| `--regular-mask-key` | `mask` | regular H5 中 mask 字段名 |

训练 patch 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-anchors` | `None` | 锚点数；空值时由观测数和 `anchor-stride` 推断 |
| `--anchor-stride` | `128` | 自动锚点数的分母，约为 `N_obs // anchor_stride` |
| `--k-patch` | `256` | 每个 anchor 选择的观测上下文规模 |
| `--top-l` | `None` | 候选邻域大小；空值时为 `k_patch + 128` |
| `--num-query` | `8` | 训练池中 query 构造提示 |
| `--pool-size` | `None` | 训练池宽度；空值时由 sampler 自动决定 |
| `--beta` | `0.3` | `diverse_topk` 多样性权重 |
| `--metric-weights` | `1.0,1.0,0.5,0.5` | 坐标加权距离 |
| `--train-anchor-selector` | `value_based_anchor_sampling` | 可选 `farthest_point_sampling`、`facility_location_anchor_sampling`、`value_based_anchor_sampling` |
| `--train-trusted-source` | `all` | `all` 使用全部 raw 观测；`regular_mask` 只使用能映射到 regular mask=True 的观测 |

value-based anchor 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--value-local-top-l` | `None` | value-based 局部候选规模；空值时等于 `top_l` |
| `--value-suppression` | `subtractive` | 邻域抑制方式，可选 `subtractive`、`multiplicative` |
| `--value-suppression-lambda` | `1.0` | 抑制强度 |
| `--value-score-tol` | `0.0` | 分数停止阈值 |
| `--value-knn-gpu-batch-rows` | `512` | GPU kNN 分批行数 |
| `--value-knn-gpu-device` | `cuda:0` | 训练锚点 GPU 设备 |
| `--value-knn-full-matrix-max-n` | `4096` | 小数据一次性全矩阵距离计算阈值 |
| `--train-knn-use-gpu` / `--no-train-knn-use-gpu` | `True` | 是否用 GPU 做 kNN |
| `--train-suppression-use-gpu` / `--no-train-suppression-use-gpu` | `True` | 是否用 GPU 做贪心抑制 |

推理 patch 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--block-size` | `None` | 显式 4D block 大小，4 个整数 |
| `--stride` | `None` | 显式 4D stride，4 个整数 |
| `--block-divisors` | `6 21 7 5` | 未提供 `block-size` 时，用 grid 维度除以该值 |
| `--stride-divisors` | `6 21 7 5` | 未提供 `stride` 时，用 grid 维度除以该值 |
| `--on-grid-collision` | `raise` | 规则网格 4D cell 冲突处理；可选 `raise`、`last` |
| `--query-mask-mode` | `regular_true` | 可选 `regular_true`、`regular_false`、`all`、`none` |
| `--infer-obs-valid-source` | `none` | 推理 context 是否受 regular mask 过滤 |
| `--infer-top-l` | `None` | 推理 context 候选邻域；空值时为 `k_patch * 2` |
| `--max-query-per-patch` | `64` | 单个推理 patch 最大 query 数 |
| `--gpu-query-chunk-size` | `64` | GPU 推理候选搜索的 query 分批数 |
| `--infer-gpu-device` | `cuda:0` | 推理候选搜索 GPU |
| `--infer-use-gpu` / `--no-infer-use-gpu` | `False` | 是否用 GPU 做推理 context 搜索 |
| `--require-full-query-coverage` / `--no-require-full-query-coverage` | `True` | 是否要求 query 全覆盖 |
| `--greedy-fill-uncovered` / `--no-greedy-fill-uncovered` | `True` | 是否用贪心策略补齐未覆盖 query |

开关参数：

| 参数 | 说明 |
| --- | --- |
| `--skip-train` | 不生成 `train_pool_idx_2d.npz` |
| `--skip-infer` | 不生成 `infer_query_context.npz` |
| `--save-legacy-anchor-files` | 额外保存旧格式 anchor 文件 |
| `--save-grid-index-map` / `--no-save-grid-index-map` | 是否保存 `grid_index_map_4d.npy` |
| `--summary-json` | 自定义 summary JSON 路径 |

主要输出：

| 文件 | 说明 |
| --- | --- |
| `train_pool_idx_2d.npz` | 训练池，包含 `pool_idx_2d` 和 `anchor_idx` |
| `infer_query_context.npz` | 推理索引，包含 `grid_query_idx_list`、`context_idx_list`、`block_id`、`block_center_grid_idx`、`anchor_grid_idx_list` |
| `infer_query_context_stats.npz` | 每个推理 patch 的 query/context 数量和缺失率统计 |
| `coord_norm_stats.npz` | 坐标归一化统计 |
| `coord_obs_norm.npy`, `coord_grid_norm.npy` | 归一化后的观测和规则网格坐标 |
| `grid_index_map_4d.npy` | 4D 逻辑网格到 regular H5 行号的映射 |
| `precompute_anchor_patch_v2_summary.json` | 预计算配置和校验摘要 |

### 6.3 shell wrapper: `tool/reg_tool/run_precompute.sh`

命令示例：

```bash
RAW_H5=/path/to/field1031_irregular.h5 \
REGULAR_H5=/path/to/field1031_label.h5 \
TARGET_H5=/path/to/field1031_mask.h5 \
PATCH_DIR=/path/to/anchor_patch_v2 \
bash tool/reg_tool/run_precompute.sh
```

常用环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BASE_DIR` | `/data/shared/测试数据/h5` | 数据根目录 |
| `RAW_H5` | `${BASE_DIR}/field1031_irregular.h5` | raw H5 |
| `REGULAR_H5` | `${BASE_DIR}/field1031_label.h5` | regular H5 |
| `TARGET_H5` | `${BASE_DIR}/field1031_mask.h5` | target/mask H5 |
| `GROUP_KEY` | `1551` | H5 group |
| `PATCH_DIR` | `${BASE_DIR}/anchor_patch_v2` | 输出目录 |
| `NUM_ANCHORS` | `7896` | 锚点数 |
| `K_PATCH` | `256` | patch context 规模 |
| `TOP_L` | `512` | 候选邻域 |
| `NUM_QUERY` | `8` | query 数提示 |
| `BETA` | `0.3` | 多样性权重 |
| `BLOCK_DIVISORS` | `6,21,7,5` | 推理 block divisors |
| `MAX_QUERY_PER_PATCH` | `128` | 推理 patch 最大 query 数 |

注意：当前 `run_precompute.sh` 默认参数列表中固定加入了 `--skip-infer`，因此默认只生成训练池，不生成 `infer_query_context.npz`。若需要推理索引，建议直接运行 `precompute_anchor_patch_v2.py` 或使用 `core.py anchor_patch` 且不要传 `--skip-infer`。

### 6.4 通用入口: `tool/reg_tool/core.py`

`core.py` 支持多种模式：

| 模式 | 命令 | 说明 |
| --- | --- | --- |
| `anchor_patch` | `python tool/reg_tool/core.py anchor_patch ...` | 推荐主线，生成训练池和 4D block 推理索引 |
| `binning` | `python tool/reg_tool/core.py binning ...` | 按复合 key 将 raw 对齐到 regular，生成 target H5 和 mask |
| `binning+csg` | `python tool/reg_tool/core.py binning+csg ...` | 先 binning，再按共炮点道集生成索引 |
| `binning+crg` | `python tool/reg_tool/core.py binning+crg ...` | 先 binning，再按共检波点道集生成索引 |
| `kdtree` | `python tool/reg_tool/core.py kdtree ...` | 基于 KDTree 的邻域覆盖 |
| `csg` | `python tool/reg_tool/core.py csg ...` | 共炮点道集索引 |
| `crg` | `python tool/reg_tool/core.py crg ...` | 共检波点道集索引 |

示例：

```bash
python tool/reg_tool/core.py anchor_patch \
  --raw_h5 /path/to/field1031_irregular.h5 \
  --regular_h5 /path/to/field1031_label.h5 \
  --target_h5 /path/to/field1031_mask.h5 \
  --patch-dir /path/to/patch_anchor_patch \
  --enable-auto-params
```

重要参数与 `precompute_anchor_patch_v2.py` 基本一致，但名称使用下划线风格，例如 `--raw_h5`、`--regular_h5`、`--k_patch`、`--top_l`、`--metric_weights`。

补充参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--raw_key_aggregate` | `mean` | `anchor_patch` 模式下是否按复合 key 对 raw 观测道求均值聚合；可选 `none`、`mean` |
| `--enable-auto-params` | 关闭 | 根据观测系统自动覆盖 `num_anchors`、`k_patch`、`top_l`、`num_query`、block divisors 等 |
| `--auto-params-anchor-stride` | `128` | 自动估计锚点数量时的步长 |
| `--trusted_mask_key` | `None` | raw H5 中用于筛选可信观测的 mask 字段 |

### 6.5 shell wrapper: `tool/reg_tool/run_core.sh`

命令示例：

```bash
RAW_H5=/path/to/field1031_irregular.h5 \
REGULAR_H5=/path/to/field1031_label.h5 \
TARGET_H5=/path/to/field1031_mask.h5 \
PATCH_DIR=/path/to/patch_anchor_patch \
ENABLE_AUTO_PARAMS=true \
SKIP_INFER=false \
bash tool/reg_tool/run_core.sh anchor_patch
```

环境变量与 `core.py` 参数一一对应。注意脚本中 `ENABLE_AUTO_PARAMS` 的默认字符串为 `fasle`，只有显式设置为 `true` 才会传入 `--enable-auto-params`。

## 七、工作流 3: 模型训练

训练主入口是：

```bash
bash run_train.sh
```

推荐显式指定路径：

```bash
NUM_GPUS=1 \
SEGY_CONFIG=field1031 \
H5_FILE=/path/to/field1031_irregular.h5 \
H5_FILE_REGULAR=/path/to/field1031_label.h5 \
DATASET_NEIGHBORS_TRAIN=/path/to/anchor_patch_v2/train_pool_idx_2d.npz \
BATCH_SIZE=8 \
EPOCHS=200 \
TRAIN_NUM_QUERY=32 \
TRACE_PS=128 \
TIME_PS=1256 \
bash run_train.sh
```

也可以直接调用：

```bash
python train.py \
  --h5File /path/to/field1031_irregular.h5 \
  --h5File_regular /path/to/field1031_label.h5 \
  --dataset_neighbors_train /path/to/train_pool_idx_2d.npz \
  --batch_size 8 \
  --epochs 200 \
  --train_num_query 32 \
  --trace_ps 128 \
  --time_ps 1256
```

### 7.1 `run_train.sh` 环境变量

| 环境变量 | 默认值 | 对应 CLI | 说明 |
| --- | --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 无 | 可见 GPU |
| `NUM_GPUS` | `4` | 启动方式 | 大于 1 时用 `accelerate launch` |
| `MODEL_NAME` | `trace_axis` | `--model_name` | 结果目录名称组成部分 |
| `BATCH_SIZE` | `8` | `--batch_size` | 每 GPU batch size |
| `LR` | `1e-4` | `--lr` | AdamW 学习率 |
| `EPOCHS` | `200` | `--epochs` | 训练轮数 |
| `SEED` | `515` | `--seed` | 随机种子 |
| `DATA_TYPE` | `df_field1031_5d` | `--data_type` | 结果目录名称组成部分 |
| `GEOM_MODE` | `relative` | `--geom_mode` | 几何条件模式 |
| `USE_MISSING_EMBEDDING` | `false` | `--use_missing_embedding` | 是否启用缺失位置适配分支 |
| `USE_PHYS_OMEGA` | `true` | `--use_phys_omega` | 是否根据物理网格步长设置 RoPE 频率 |
| `USE_P_SCALE` | `false` | `--use_p_scale` | 是否按 `p_scale` 缩放坐标统计 |
| `H5_DIR` | `/data/shared/测试数据/h5` | 无 | 默认数据目录 |
| `H5_FILE` | `${H5_DIR}/field1031_irregular.h5` | `--h5File` | irregular H5 |
| `H5_FILE_REGULAR` | `${H5_DIR}/field1031_label.h5` | `--h5File_regular` | regular H5 |
| `DATASET_NEIGHBORS_TRAIN` | `${H5_DIR}/anchor_patch_v2/train_pool_idx_2d.npz` | `--dataset_neighbors_train` | 训练索引 |
| `TRAIN_NUM_QUERY` | `32` | `--train_num_query` | 每个训练样本 query 道数 |
| `TRAIN_CONTEXT_SIZE` | 空 | `--train_context_size` | 固定 context 数；空值时为 `trace_ps - train_num_query` |
| `PATCH_BETA` | `0.3` | `--patch_beta` | context 选择多样性权重 |
| `FORCE_ANCHOR_QUERY` | `false` | `--force_anchor_query` | 是否强制 anchor 进入 query |
| `TIME_PS` | `1256` | `--time_ps` | 每道时间采样点数 |
| `TRACE_PS` | `128` | `--trace_ps` | 每个 patch 总道数 |
| `SEGY_CONFIG` | `field1031` | 环境变量 | SEG-Y preset |

### 7.2 `train.py` 参数

数据参数来自 `config/data_config.py`：

| 参数 | Python 默认值 | 说明 |
| --- | --- | --- |
| `--h5File` | `dongfang_field1031/raw5d_data1104.h5` | irregular H5 |
| `--h5File_regular` | `dongfang_field1031/reg5dbin_label1031.h5` | regular H5 |
| `--time_ps` | `1256` | 时间采样点数，长道左裁剪，短道左补零 |
| `--trace_ps` | `128` | patch 总道数 |
| `--dataset_neighbors_train` | `None` | 训练 `train_pool_idx_2d.npz` |
| `--dataset_neighbors_test` | `None` | 兼容保留参数 |
| `--train_num_query` | `16` | query 道数；`run_train.sh` 默认覆盖为 32 |
| `--train_context_size` | `None` | 固定 context 数 |
| `--patch_beta` | `0.3` | 多样性权重 |
| `--force_anchor_query` | `False` | 是否强制 anchor 进入 query |
| `--trace_sort_keys` | `offset,azimuth` | patch 内 trace 排序键；当前 `train.py` 构造数据集时该参数未实际传入 |
| `--epoch_repeat` | `4` | 每个 anchor 在一个 epoch 内重复采样次数 |
| `--use_phys_omega` | `False` | 是否启用物理 RoPE 频率；`run_train.sh` 默认覆盖为 true |
| `--use_p_scale` | `True` | 是否缩放坐标统计；`run_train.sh` 默认覆盖为 false |
| `--dataset_type` | `queryctx` | 当前训练脚本固定使用 `DatasetH5_all_queryctx` |

训练参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_name` | `trace_axis` | 结果目录名称 |
| `--batch_size` | `2` | 每 GPU batch size；shell 默认覆盖为 8 |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--epochs` | `200` | 训练 epoch |
| `--seed` | `515` | 随机种子 |
| `--data_type` | `df_field1031_5d` | 结果目录名称 |
| `--geom_mode` | `relative` | 可选 `source`、`receiver`、`relative` |
| `--use_missing_embedding` | `False` | 是否启用 missing focus adapter |
| `--pe_type` | `transformer` | 位置编码类型参数，传入模型 |
| `--rope_base` | `10000.0` | RoPE base；当 `use_phys_omega=true` 时会被自动物理频率覆盖 |
| `--path_type` | `Linear` | Flow path，可选 `Linear`、`GVP`、`VP` |
| `--prediction` | `velocity` | 预测目标，可选 `velocity`、`score`、`noise` |
| `--loss_weight` | `None` | 可选 `velocity`、`likelihood` |
| `--sampling_method` | `ode` | 采样方式，可选 `ode`、`sde` |
| `--ode_num_steps` | `50` | ODE 采样步数 |
| `--sde_num_steps` | `250` | SDE 采样步数 |
| `--results_dir` | `./resultsFPM` | 输出根目录 |
| `--save_every` | `10` | checkpoint 保存间隔 |
| `--accumulation_steps` | `4` | 梯度累积步数 |

### 7.3 训练内部逻辑

`train.py` 固定使用以下核心配置：

| 模块 | 配置 |
| --- | --- |
| 数据集 | `DatasetH5_all_queryctx(train=True)` |
| 模型 | `SeisDiTRopeV2(image_channels=2, n_channels=32, num_layers=8, d_model=512)` |
| 优化器 | `AdamW(betas=(0.9, 0.95), weight_decay=1e-4)` |
| 精度 | `Accelerator(mixed_precision="fp16")` |
| 梯度裁剪 | `max_norm=1.0` |
| 学习率 | 前 5 个 epoch warmup，之后 `CosineAnnealingLR(eta_min=5e-5)` |
| 验证 | 第 0 个 epoch 和每 10 个 epoch，最多 50 个 batch |

输出目录格式：

```text
resultsFPM/<model_name>_datatype_<data_type>_queryctx/
├── checkpoints/
│   └── model-*.pth
├── logs/
│   ├── training_config.json
│   └── training_log.txt
└── images/
```

`training_config.json` 会保存训练参数、数据路径、模型参数量、坐标统计和 RoPE 频率信息，推理时会尝试从 checkpoint 所在目录自动读取。

## 八、工作流 4: 推理与 SEG-Y 回填

推理主入口：

```bash
bash run_infer.sh
```

推荐显式指定路径：

```bash
NUM_GPUS=1 \
SEGY_CONFIG=field1031 \
CHECKPOINT=/path/to/resultsFPM/.../checkpoints/model-20.pth \
H5_IRREGULAR=/path/to/field1031_irregular.h5 \
H5_REGULAR=/path/to/field1031_label.h5 \
H5_MASK=/path/to/field1031_mask.h5 \
MASK_SEGY=/path/to/mask_from_label.sgy \
DATASET_NEIGHBORS_INFER=/path/to/anchor_patch_v2/infer_query_context.npz \
OUTPUT_DIR=/path/to/gen_fill_results \
bash run_infer.sh
```

直接调用：

```bash
python infer_cli.py \
  --checkpoint /path/to/model-20.pth \
  --segy_config field1031 \
  --h5_irregular /path/to/field1031_irregular.h5 \
  --h5_regular /path/to/field1031_label.h5 \
  --h5_mask /path/to/field1031_mask.h5 \
  --mask_path /path/to/mask_from_label.sgy \
  --dataset_neighbors_infer /path/to/infer_query_context.npz \
  --output_dir /path/to/gen_fill_results \
  --batch_size 18 \
  --time_ps 1256 \
  --trace_ps 128 \
  --strict_fill
```

### 8.1 `run_infer.sh` 环境变量

| 环境变量 | 默认值 | 对应 CLI | 说明 |
| --- | --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 无 | 可见 GPU |
| `NUM_GPUS` | `8` | 启动方式 | 大于 1 时使用 `torchrun` |
| `MASTER_PORT` | `29502` | `torchrun` | 分布式端口 |
| `CHECKPOINT` | `/data/shared/测试数据/h5/model-20.pth` | `--checkpoint` | 模型权重 |
| `H5_DIR` | `/data/shared/测试数据/h5` | 无 | 默认数据目录 |
| `H5_IRREGULAR` | `${H5_DIR}/field1031_irregular.h5` | `--h5_irregular` | irregular H5 |
| `H5_REGULAR` | `${H5_DIR}/field1031_label.h5` | `--h5_regular` | regular H5 |
| `H5_MASK` | `${H5_DIR}/field1031_mask.h5` | `--h5_mask` | mask H5 |
| `MASK_SEGY` | `/data/shared/测试数据/mask_from_label.sgy` | `--mask_path` | 回填模板 SEG-Y |
| `DATASET_NEIGHBORS_INFER` | `${H5_DIR}/patchV4/infer_query_context.npz` | `--dataset_neighbors_infer` | 推理索引 |
| `LABEL_SEGY` | 空 | `--label_segy` | 可选标签 SEG-Y，用于 residual |
| `OUTPUT_DIR` | `gen_fill_results_v2` | `--output_dir` | 输出目录 |
| `OUTPUT_SEGY` | `${OUTPUT_DIR}/filled_missing.sgy` | `--output_segy` | 补全 SEG-Y |
| `OUTPUT_RESIDUAL_SEGY` | `${OUTPUT_DIR}/residual.sgy` | `--output_residual_segy` | residual SEG-Y |
| `BATCH_SIZE` | `18` | `--batch_size` | 推理 batch size |
| `TIME_PS` | `1256` | `--time_ps` | 时间采样点 |
| `TRACE_PS` | `128` | `--trace_ps` | patch 总道数 |
| `HEADER_MODE` | `fixed` | `--header_mode` | SEG-Y key 读取方式 |
| `GEOM_MODE` | `relative` | `--geom_mode` | 几何模式，需与训练一致 |
| `USE_MISSING_EMBEDDING` | `false` | `--use_missing_embedding` | 需与训练一致 |
| `USE_P_SCALE` | `false` | `--use_p_scale` | 会被 checkpoint 的 `training_config.json` 覆盖 |
| `USE_PHYS_OMEGA` | `true` | `--use_phys_omega` | 是否使用物理 RoPE 频率 |
| `SAMPLING_METHOD` | `ode` | `--sampling_method` | `ode` 或 `sde` |
| `ODE_NUM_STEPS` | `50` | `--ode_num_steps` | ODE 步数 |
| `ODE_SAMPLING_METHOD` | `dopri5` | `--ode_sampling_method` | ODE 求解器 |
| `ODE_ATOL` | `1e-6` | `--ode_atol` | ODE 绝对误差 |
| `ODE_RTOL` | `1e-3` | `--ode_rtol` | ODE 相对误差 |
| `SDE_NUM_STEPS` | `250` | `--sde_num_steps` | SDE 步数 |
| `VISUALIZE` | `true` | `--visualize` | 是否保存样本可视化 |
| `VIS_BATCHES` | `0` | `--vis_batches` | 可视化样本数，0 表示全部 |
| `SORT_SEGY` | `true` | `--sort_segy` | 是否输出排序后的 SEG-Y |
| `SEGY_CONFIG` | `field1031` | `--segy_config` | SEG-Y preset |

`run_infer.sh` 固定传入 `--strict_fill`。如果补全后仍有未填充缺失道、已有道被改动，程序会抛错。

### 8.2 `infer_cli.py` 参数

必需输入：

| 参数 | 说明 |
| --- | --- |
| `--checkpoint` | 训练得到的 `.pth` 权重 |
| `--h5_irregular` | irregular H5，提供 context 道 |
| `--h5_regular` | regular H5，提供 query 道和坐标统计 |
| `--h5_mask` | mask H5，目前 CLI 参数保留，主要路径由数据集和 SEG-Y 回填使用 |
| `--mask_path` | 回填模板 SEG-Y，缺失道通常为全零 |
| `--dataset_neighbors_infer` | `infer_query_context.npz` |

输出参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output_dir` | `gen_fill_results` | 输出目录 |
| `--output_segy` | `<output_dir>/filled_missing.sgy` | 补全 SEG-Y |
| `--output_residual_segy` | `<output_dir>/residual.sgy` | residual SEG-Y |
| `--label_segy` | `None` | 标签 SEG-Y；提供时生成 residual |

模型与采样参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `cuda:0` | 单 GPU/CPU 设备；多 GPU 时由 `LOCAL_RANK` 决定 |
| `--batch_size` | `6` | 直接调用默认值；shell 默认覆盖为 18 |
| `--time_ps` | `1256` | 时间采样点；若 checkpoint 中有训练配置，会优先使用训练配置 |
| `--trace_ps` | `128` | patch 总道数 |
| `--missing_eps` | `1e-10` | 判断全零缺失道的阈值 |
| `--header_mode` | `fixed` | `fixed` 或 `self_computed` |
| `--strict_load` | 默认严格加载 | 该参数为 `store_false`，传入后会关闭严格加载 |
| `--strict_fill` | `False` | 严格校验回填完整性 |
| `--geom_mode` | `relative` | 需与训练一致 |
| `--use_missing_embedding` | `False` | 需与训练一致 |
| `--use_p_scale` | `False` | 会被训练配置覆盖 |
| `--use_phys_omega` | `True` | 使用物理 RoPE 频率 |
| `--model_type` | `trace_axis` | 保留参数 |
| `--pe_type` | `transformer` | 传入模型 |
| `--path_type` | `Linear` | `Linear`、`GVP`、`VP` |
| `--prediction` | `velocity` | `velocity`、`score`、`noise` |
| `--loss_weight` | `None` | Flow loss 权重类型 |
| `--sampling_method` | `ode` | `ode` 或 `sde` |
| `--ode_sampling_method` | `dopri5` | ODE 求解器 |
| `--ode_num_steps` | `50` | ODE 步数 |
| `--ode_atol` | `1e-6` | ODE 绝对误差 |
| `--ode_rtol` | `1e-3` | ODE 相对误差 |
| `--sde_sampling_method` | `Euler` | SDE 求解器 |
| `--sde_num_steps` | `250` | SDE 步数 |
| `--segy_config` | `None` | SEG-Y preset；空值时使用环境变量或默认 preset |
| `--sort_segy` | `False` | 是否额外输出 `_sorted.sgy` |
| `--visualize` | `False` | 是否保存可视化 PNG |
| `--vis_batches` | `0` | 可视化样本数，0 表示全部 |

### 8.3 推理输出

| 文件 | 说明 |
| --- | --- |
| `filled_missing.sgy` | 将预测道写回 mask SEG-Y 的结果 |
| `filled_missing_sorted.sgy` | 当 `--sort_segy true` 时额外输出 |
| `residual.sgy` | 提供 `--label_segy` 时输出，写入预测与标签差值 |
| `summary.json` | 回填数量、未填充数量、预测 key 数、耗时等 |
| `filled_missing_keys.csv` | 成功写入的缺失道 key |
| `unfilled_missing_keys.csv` | 未被写入的缺失道 key |
| `still_missing_after_write_keys.csv` | 写入后仍接近零的缺失道 |
| `observed_changed_keys.csv` | 原已有道被改动的列表 |
| `unmatched_prediction_keys.csv` | 预测 key 未匹配到 SEG-Y 道头的列表 |
| `infer.log` | 推理日志 |
| `infer.stdout.log` | `run_infer.sh` 捕获的 stdout 日志 |
| `vis/sample_*.png` | 可视化图，包含 masked input、prediction、ground truth、residual |

## 九、数据集与索引格式

### 9.1 `DatasetH5_all_queryctx`

实现位置：`dataset/dataset_reg.py`。

该数据集是训练和推理主数据集，使用两份 H5：

| 输入 | 用途 |
| --- | --- |
| `h5File` | irregular H5，提供观测 context 道 |
| `h5File_regular` | regular H5，提供规则网格 query 道、坐标范围和 key |
| `dataset_neighbors` | 训练或推理 `.npz` 索引文件 |

训练模式识别 `train_pool_idx_2d.npz`：

```text
pool_idx_2d: [N_anchor, pool_width]
anchor_idx: [N_anchor]
```

训练采样逻辑：

1. 从 `pool_idx_2d` 取一个观测池。
2. 随机选 `train_num_query` 个 query 道。
3. 在剩余候选中使用 `diverse_topk` 选择 context 道。
4. query 道在 `masked_patch` 中置零。
5. 按 `trace_sort_keys` 对 patch 内 trace 排序。
6. 对振幅按观测道 99.5 百分位裁剪并归一化。

推理模式识别 `infer_query_context.npz`：

```text
grid_query_idx_list: regular H5 中待预测 query 行号
context_idx_list: irregular H5 中作为上下文的观测行号
block_id: block 编号
block_center_grid_idx: block 中心行号
anchor_grid_idx_list: anchor 行号列表
```

推理采样逻辑：

1. query 数据从 regular H5 读取。
2. context 数据从 irregular H5 读取。
3. query 在输入中置零。
4. 推理预测只对 `is_query=True` 的道累计。
5. 预测结果按 `shot_line, shot_stake, recv_line, recv_stake` key 汇总。

### 9.2 `DatasetH5Interp`

实现位置：`dataset/dataset_interp.py`。

该数据集使用单 H5，依赖 H5 内部 `mask` 字段标记已有和缺失位置。当前 `train.py` 没有直接启用该数据集，但保留了与 `core.py` 输出索引兼容的训练和推理逻辑。

### 9.3 振幅与时间处理

| 处理项 | 规则 |
| --- | --- |
| 时间长度 | 若原始 trace 长于 `time_ps`，保留最后 `time_ps` 个采样；若短于 `time_ps`，左侧补零 |
| 振幅 scale | 使用 context/观测道绝对值 99.5 百分位作为裁剪和归一化阈值 |
| 推理回填长度 | `infer.py::fit_trace` 会按原 SEG-Y trace sample 数恢复长度 |

## 十、模型与 Flow Matching

### 10.1 模型结构

当前训练脚本使用 `model/seisdit_trace_axis.py` 中的 `SeisDiTRopeV2`。

核心结构：

```text
输入 [x_t, x_cond]，形状 [B, 2, trace, time]
  |
  | tokenizer: query/noisy 通道和条件通道分别 Conv2d，再 1x1 fuse
  v
Encoder: Resblock + AdaTimeModulation + time-axis downsample
  |
  v
Bottleneck:
  - DiTBlockTrace
  - TraceAxisAttention2D
  - SegmentedRoPEExpCached
  - geom MLP + adaLN modulation
  |
  v
Decoder: skip connection + time-axis upsample
  |
  v
输出 velocity/score/noise 预测，默认 velocity
```

几何条件：

| 参数 | 说明 |
| --- | --- |
| `geom_mode=source` | 使用 source 坐标作为几何 MLP 输入 |
| `geom_mode=receiver` | 使用 receiver 坐标作为几何 MLP 输入 |
| `geom_mode=relative` | 使用 `(sx - rx, sy - ry)` 作为几何 MLP 输入 |
| `use_phys_omega=true` | 根据规则网格步长按 Nyquist 思路估计 RoPE 物理频率 |

### 10.2 Flow Matching

实现位置：`fpm.py`、`transport/transport.py`、`transport/path.py`。

默认配置：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `path_type` | `Linear` | 使用 `ICPlan`，即线性路径 |
| `prediction` | `velocity` | 模型预测速度场 |
| `sampling_method` | `ode` | 推理时用 ODE 积分 |
| `ode_sampling_method` | `dopri5` | ODE solver |
| `ode_num_steps` | `50` | ODE 步数 |

训练损失：

```text
x_t = t * x1 + (1 - t) * x0
u_t = x1 - x0
loss = MSE(model(x_t, t, cond), u_t)
```

当 `prediction=score` 或 `prediction=noise` 时，`transport/transport.py` 会根据 path 的 drift/sigma 变换目标并应用对应 loss weight。

## 十一、常见问题与建议补充

| 问题 | 说明与处理 |
| --- | --- |
| 多 GPU 训练找不到 `accelerate_config.yaml` | 当前仓库未包含该文件。使用 `NUM_GPUS=1`，或自行生成 `accelerate_config.yaml` 后再多卡训练 |
| `run_precompute.sh` 没有生成 `infer_query_context.npz` | 该 wrapper 当前固定追加 `--skip-infer`。需要推理索引时请直接运行 `precompute_anchor_patch_v2.py` 或 `core.py anchor_patch` |
| `run_core.sh` 设置了 `ENABLE_AUTO_PARAMS` 但未生效 | 只有 `ENABLE_AUTO_PARAMS=true` 才会传入 `--enable-auto-params` |
| 回填后仍有未填充道 | 查看 `summary.json`、`unfilled_missing_keys.csv` 和 `unmatched_prediction_keys.csv`，通常是 query key 与 SEG-Y header key 不一致 |
| `strict_fill failed` | `run_infer.sh` 默认传入 `--strict_fill`，任何未填充缺失道或已有道被改动都会报错 |
| `segc3` 数据没有 line/stake 道头 | 使用 `--mode self_computed` 和 `SEGY_CONFIG=segc3`，复合 key 由缩放后的坐标计算 |
| 训练和推理结果异常 | 核对 `geom_mode`、`use_missing_embedding`、`use_p_scale`、`time_ps`、`trace_ps` 是否与训练一致 |
| H5 中没有 `mask` 字段 | `core.py` 和 `precompute_anchor_patch_v2.py` 可尝试从 `target_h5` 的零值道推断 mask |

建议补充：

| 项目 | 建议 |
| --- | --- |
| 数据路径模板 | 当前 shell 脚本内默认路径均指向本机或服务器示例路径，建议项目交付时补充一份实际数据目录规范 |
| `accelerate_config.yaml` | 如需多 GPU 训练，建议将可复用配置纳入仓库或文档 |
| mask 语义 | 建议在数据交付说明中明确 `mask=True/1` 表示已有道还是待预测道；当前预计算脚本的 `query-mask-mode` 可以切换 `regular_true` 和 `regular_false` |
| SEG-Y key 唯一性 | 建议在验收前统计 regular H5 复合 key 是否唯一，避免回填时一键多道或未匹配 |

## 十二、端到端命令模板

以下模板展示从三份 SEG-Y 到补全 SEG-Y 的最小主线。

### 12.1 转换

```bash
python tool/convert_tool/Segy2H5.py \
  --irr /data/project/raw/irregular.sgy \
  --mask /data/project/raw/mask.sgy \
  --label /data/project/raw/label.sgy \
  --dataset-name field1031 \
  --mode fixed \
  --config field1031
```

### 12.2 预计算

```bash
python tool/reg_tool/core.py anchor_patch \
  --raw_h5 /data/project/raw/h5/field1031_irregular.h5 \
  --regular_h5 /data/project/raw/h5/field1031_label.h5 \
  --target_h5 /data/project/raw/h5/field1031_mask.h5 \
  --group_key 1551 \
  --patch-dir /data/project/raw/h5/anchor_patch_v2 \
  --enable-auto-params
```

### 12.3 训练

```bash
NUM_GPUS=1 \
SEGY_CONFIG=field1031 \
H5_FILE=/data/project/raw/h5/field1031_irregular.h5 \
H5_FILE_REGULAR=/data/project/raw/h5/field1031_label.h5 \
DATASET_NEIGHBORS_TRAIN=/data/project/raw/h5/anchor_patch_v2/train_pool_idx_2d.npz \
BATCH_SIZE=8 \
EPOCHS=200 \
TRAIN_NUM_QUERY=32 \
TRACE_PS=128 \
TIME_PS=1256 \
bash run_train.sh
```

### 12.4 推理回填

```bash
NUM_GPUS=1 \
SEGY_CONFIG=field1031 \
CHECKPOINT=/data/project/seismic_flow_5d/resultsFPM/trace_axis_datatype_df_field1031_5d_queryctx/checkpoints/model-20.pth \
H5_IRREGULAR=/data/project/raw/h5/field1031_irregular.h5 \
H5_REGULAR=/data/project/raw/h5/field1031_label.h5 \
H5_MASK=/data/project/raw/h5/field1031_mask.h5 \
MASK_SEGY=/data/project/raw/mask.sgy \
DATASET_NEIGHBORS_INFER=/data/project/raw/h5/anchor_patch_v2/infer_query_context.npz \
OUTPUT_DIR=/data/project/output/gen_fill_results \
BATCH_SIZE=18 \
bash run_infer.sh
```
