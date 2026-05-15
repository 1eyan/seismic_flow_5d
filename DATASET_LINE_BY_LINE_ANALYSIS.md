# DatasetH5_all_queryctx 逐行分析

## 文件结构概览

```
DatasetH5_all_queryctx
├── __init__           # 初始化：加载H5、元数据、计算坐标统计
├── 公开方法           # typical_grid_step, __len__
├── H5加载             # _load_h5_group
├── 元数据加载          # _load_patch_metadata (核心！)
├── 索引工具            # _index_row, _take_rows
├── 时间/缩放工具       # _crop_or_pad_time, _time_axis_2d, _scale_pair
├── 采样工具            # _sample_rng
├── 轨迹排序            # _sort_traces
├── 坐标归一化          # _normalize_coords, compute_coord_stats
├── 训练样本构建        # _build_train_query_context_sample (核心！)
├── 推理样本构建        # _build_infer_query_context_sample (核心！)
└── 调度器              # __getitem__ (根据模式分发)
```

---

## 详细逐行分析

### 第 1-12 行：类定义与文档字符串

```python
class DatasetH5_all_queryctx:
    """Query-context dataset for seismic 5D interpolation.

    Supports two metadata modes:
    - ``train_pool``: per-sample pool of trace indices; online query/context selection
    - ``infer_query_context``: precomputed grid-query + context index pairs

    Attributes:
        h5File: path to irregular (observed) H5 file
        h5File_regular: path to regular-grid H5 file
        time_ps: number of time samples per trace
        trace_ps: total traces per patch (query + context)
        coord_stats: per-axis min/max and grid-step statistics
        patch_mode: "train_pool" or "infer_query_context"
    """
```

**关键概念**：
- 类名带后缀 `all_queryctx` 表示"处理所有类型的query-context分割"
- 支持两种模式（通过元数据自动识别）：
  - **train_pool**：每个样本有一个候选trace pool，运行时随机选query/context
  - **infer_query_context**：query/context索引预计算好，推理时固定使用
- `h5File` vs `h5File_regular`：不规则观测 vs 规则网格（关键区分！）

---

### 第 13-14 行：类属性

```python
    _COORD_COL = _COORD_COL  # from segy_config
```

**作用**：
- 从 `segy_config` 导入的静态映射，定义坐标列名到H5数据集列的索引
- 例如：`{"sx": 0, "sy": 1, "rx": 2, "ry": 3}` （假设）
- 用于后续 `_sort_traces()` 中的 `coords_patch[:, self._COORD_COL[k]]` 查找

---

### 第 16-48 行：`__init__()` 方法 - 初始化签名

```python
    def __init__(
        self,
        h5File=None,                          # ① 观测H5路径
        h5File_regular=None,                  # ② 规则网格H5路径
        h5File_tgt=None,                      # ③ 目标H5路径（可选）
        dataset_neighbors=None,               # ④ 关键！元数据.npz路径
        train=None,                           # ⑤ 训练/推理标志
        train_num_query: int = 16,            # ⑥ 每个样本抽多少query traces
        train_context_size: Optional[int] = None,  # ⑦ 每个样本抽多少context traces
        patch_beta: float = 0.3,              # ⑧ diverse_topk多样性权重
        patch_metric_weights=None,            # ⑨ 4D坐标距离度量权重 [sx,sy,rx,ry]
        force_anchor_query: bool = False,     # ⑩ 强制锚点必须作为query
        trace_sort_keys: Tuple[str, ...] = TRACE_SORT_KEYS,  # ⑪ 排序维度
        use_p_scale: bool = False,            # ⑫ 是否应用p_scale坐标缩放
        time_ps: int = 1256,                  # ⑬ 每条trace时间采样点数
        trace_ps: int = 128,                  # ⑭ 每个patch最多trace数
    ):
```

**参数作用详表**：

| 参数 | 类型 | 默认值 | 作用 | 关键性 |
|------|------|--------|------|--------|
| `h5File` | str | None | 观测数据H5 | ★★★ |
| `h5File_regular` | str | None | 规则网格H5 | ★★★ |
| `h5File_tgt` | str | None | 目标数据H5 | ★☆☆ |
| `dataset_neighbors` | str | None | 元数据.npz（train_pool_idx_2d.npz等） | ★★★ |
| `train` | bool | None | True→训练，False→推理 | ★★★ |
| `train_num_query` | int | 16 | 训练样本中query数量（随机抽样） | ★★☆ |
| `train_context_size` | int | None | 训练样本中context数量（None→自动 trace_ps-query） | ★★☆ |
| `patch_beta` | float | 0.3 | diverse_topk中距离vs多样性权衡 | ★☆☆ |
| `patch_metric_weights` | list | None | 4D距离度量权重 | ★☆☆ |
| `force_anchor_query` | bool | False | 如果锚点在pool中，强制选为query | ★☆☆ |
| `trace_sort_keys` | tuple | TRACE_SORT_KEYS | 按哪些维度排序traces | ★☆☆ |
| `use_p_scale` | bool | False | 坐标缩放开关 | ★☆☆ |
| `time_ps` | int | 1256 | 时间轴采样点 | ★★☆ |
| `trace_ps` | int | 128 | patch最大宽度 | ★★☆ |

---

### 第 50-51 行：`__init__()` - 初始化开始

```python
        super().__init__()
        print("Loading dataset...")
```

**分析**：
- `super().__init__()` 调用父类（通常是 `object` 或 PyTorch `Dataset`）
- 打印消息用于调试，表示开始加载

---

### 第 53-68 行：`__init__()` - 基本参数存储

```python
        self.h5File = h5File
        self.h5File_regular = h5File_regular
        self.h5File_tgt = h5File_tgt
        self.time_ps = time_ps
        self.trace_ps = trace_ps
        self.train = train
        self._rng = np.random.default_rng(123)
        self.std_val = None
        self.train_num_query = int(max(1, train_num_query))
        self.train_context_size = (
            None if train_context_size is None else int(max(1, train_context_size))
        )
        self.patch_beta = float(patch_beta)
        self.patch_metric_weights = patch_metric_weights
        self.force_anchor_query = bool(force_anchor_query)
        self.trace_sort_keys = tuple(trace_sort_keys)
        self.use_p_scale = use_p_scale
```

**关键行详解**：

- **第 53-56 行**：直接保存H5文件路径和数据形状参数
  ```python
  self.h5File = h5File                    # 观测数据
  self.h5File_regular = h5File_regular    # 规则网格
  self.h5File_tgt = h5File_tgt            # 目标（可选）
  self.time_ps = time_ps                  # 每条trace 1256 个时间点
  self.trace_ps = trace_ps                # 每个patch最多 128 条traces
  ```

- **第 57-58 行**：随机数生成器初始化
  ```python
  self.train = train                      # 训练标志
  self._rng = np.random.default_rng(123)  # 固定种子，保证可重现性
  ```
  ⚠️ **注意**：种子为 `123` 是固定的，**不是** 用户传入的。这可能导致每次运行相同随机序列。

- **第 59-62 行**：参数规范化
  ```python
  self.std_val = None                     # 缓存标准差（用于归一化）
  self.train_num_query = int(max(1, train_num_query))      # ≥1
  self.train_context_size = (
      None if train_context_size is None else int(max(1, train_context_size))
  )
  ```
  💡 **含义**：`train_context_size=None` 时后续自动计算为 `max(1, trace_ps - q_eff)`

- **第 63-68 行**：训练超参
  ```python
  self.patch_beta = float(patch_beta)                    # 多样性权重
  self.patch_metric_weights = patch_metric_weights       # 可能为None，后续处理
  self.force_anchor_query = bool(force_anchor_query)     # 强制锚点约束
  self.trace_sort_keys = tuple(trace_sort_keys)          # 排序键
  self.use_p_scale = use_p_scale                         # 坐标缩放
  ```

---

### 第 70-72 行：`__init__()` - 物理常数

```python
        self.dt_ms = 4       # 时间采样间隔（毫秒）
        self.t0_ms = 0       # 起始时间（毫秒）
        self.scale = None    # 坐标缩放因子（后续计算）
```

**地球物理背景**：
- 地震勘探中，通常 4ms 是标准采样率
- `t0_ms=0` 表示从时间 0 开始记录
- `self.scale` 用于坐标标准化（见 `compute_coord_stats()` 第 776-791 行）

---

### 第 74-75 行：`__init__()` - 加载H5数据

```python
        self.h5_data = self._load_h5_group(self.h5File)
        self.h5_data_regular = self._load_h5_group(self.h5File_regular)
        self.h5_data_tgt = {}
```

**执行流**：
1. `_load_h5_group(self.h5File)` 读取观测数据H5
2. `_load_h5_group(self.h5File_regular)` 读取规则网格H5
3. 初始化空字典用于目标H5（备用）

---

### 第 77-79 行：`__init__()` - 调试打印

```python
        print(self.h5_data_regular["data"].shape)
        print(self.h5_data["data"].shape)
        print("loading data")
```

**作用**：显示两个H5的数据形状
- 例如：`(10000, 1256)` 表示 10000 条traces，每条 1256 个时间点

---

### 第 81-84 行：`__init__()` - 坐标统计与元数据加载

```python
        self.coord_stats = self.compute_coord_stats()
        print("coord_stats computed")
        self.patch_meta = self._load_patch_metadata(dataset_neighbors)
        self.patch_mode = self.patch_meta["mode"]
```

**核心逻辑**：
1. **第 81 行**：计算坐标统计（min/max/grid_step等），用于后续归一化
2. **第 83 行**：加载元数据（**最关键的一行**）
   - 读取 `.npz` 文件，解析出 pool 或 query/context 索引
   - 返回字典 `{"mode": "train_pool"|"infer_query_context", ...}`
3. **第 84 行**：提取模式标记

---

### 第 85-86 行：`__init__()` - 样本计数

```python
        print(self.patch_mode)
        self.num_samples = int(self.patch_meta["num_samples"])
        print(f"patch metadata mode: {self.patch_mode}, samples: {self.num_samples}")
```

**作用**：获取数据集样本总数，用于 `__len__()` 和 DataLoader

---

### 第 91-95 行：公开方法 - `typical_grid_step()`

```python
    def typical_grid_step(self, arr, eps=1e-9):
        u = np.sort(np.unique(arr))      # ① 唯一值排序
        if u.size < 2:
            return None, u               # ② 少于2个值 → 无法计算间隔
        d = np.diff(u)                   # ③ 相邻差值
        d = d[d > eps]                   # ④ 过滤掉小于阈值的差值
        if d.size == 0:
            return None, u               # ⑤ 所有差值都很小 → 无法计算
        return float(np.median(d)), u    # ⑥ 返回中位数作为"典型"间隔
```

**示例**：
```
arr = [1.0, 2.0, 3.0, 5.0, 6.0]
u = [1.0, 2.0, 3.0, 5.0, 6.0]
d = [1.0, 1.0, 2.0, 1.0]
median(d) = 1.0  ← 网格步长
```

**地球物理意义**：计算采样间隔，用于判断规则/不规则网格

---

### 第 97 行：公开方法 - `__len__()`

```python
    def __len__(self):
        return self.num_samples
```

**PyTorch约定**：DataLoader 调用此方法获取数据集大小

---

### 第 102-107 行：内部方法 - `_load_h5_group()`

```python
    @staticmethod
    def _load_h5_group(file_path):
        with File(file_path, "r") as f:
            for key in f:
                node = f[key]
                if hasattr(node, "keys") and "data" in node:
                    break
            return {k: node[k][:] for k in node.keys()}
```

**详细步骤**：

1. **第 103 行**：打开H5文件（只读）
   ```python
   with File(file_path, "r") as f:
   ```

2. **第 104-106 行**：寻找包含 "data" 的group
   ```python
   for key in f:
       node = f[key]
       if hasattr(node, "keys") and "data" in node:
           break
   ```
   💡 **作用**：H5通常结构为 `group1/data`, `group1/sx`, etc.
   
   这段代码寻找第一个包含 "data" 的group。例如：
   ```
   file.h5
   ├─ group_1551/
   │  ├─ data       ← 找到！
   │  ├─ sx
   │  ├─ sy
   │  └─ ...
   ```

3. **第 107 行**：读取group内所有datasets到内存
   ```python
   return {k: node[k][:] for k in node.keys()}
   ```
   例如返回：`{"data": ndarray, "sx": ndarray, "sy": ndarray, ...}`

**⚠️ 注意**：`[:]` 强制加载整个数组到内存，适合数据量不超大的情况

---

### 第 112-165 行：内部方法 - `_load_patch_metadata()` 【★★★ 核心】

这是最关键的方法，决定数据集如何运作。

```python
    def _load_patch_metadata(self, path: Optional[str]) -> Dict[str, Any]:
        if path is None:
            raise ValueError("dataset_neighbors is required")
        raw = np.load(path, allow_pickle=True)
```

**第 112-115 行**：加载.npz文件

```python
        if hasattr(raw, "files"):
            arrays = {k: raw[k] for k in raw.files}
            raw.close()
        else:
            arrays = {"0": raw}
```

**第 116-118 行**：适配两种.npz格式
- 新格式：`np.savez(file, pool_idx_2d=..., anchor_idx=...)`
  - `raw.files` 会列出所有键
  - 转为字典：`{"pool_idx_2d": ndarray, "anchor_idx": ndarray, ...}`
- 旧格式：`np.save(file, array)` 或单个数组
  - 没有 `files` 属性
  - 转为：`{"0": array}`

---

### 第 120-128 行：推理模式识别

```python
        if "grid_query_idx_list" in arrays and (
            "context_idx_list" in arrays or "patch_idx_list" in arrays
        ):
            num_samples = len(arrays["grid_query_idx_list"])
            return {
                "mode": "infer_query_context",
                "num_samples": int(num_samples),
                "grid_query_idx_list": arrays["grid_query_idx_list"],
                "context_idx_list": arrays.get(
                    "context_idx_list", arrays.get("patch_idx_list")
                ),
                "block_id": arrays.get("block_id"),
                "block_center_grid_idx": arrays.get("block_center_grid_idx"),
                "anchor_grid_idx_list": arrays.get("anchor_grid_idx_list"),
            }
```

**条件**：存在 `grid_query_idx_list` 和 `context_idx_list` 或 `patch_idx_list`

**返回**：
```python
{
    "mode": "infer_query_context",          # 推理模式标记
    "num_samples": N,                        # 总patch数
    "grid_query_idx_list": [objarr],         # 每个patch的query索引
    "context_idx_list": [objarr],            # 每个patch的context索引
    "block_id": array,                       # 块ID（可选）
    "block_center_grid_idx": array,          # 块中心（可选）
    "anchor_grid_idx_list": [objarr]         # 锚点（可选）
}
```

**对象数组说明**：
- `grid_query_idx_list` 是 shape `(N,)` 的object数组
- 每个元素是不同长度的 int64 数组
- 例如：`[array([1,2,3]), array([4,5]), array([6,7,8,9])]`

---

### 第 130-136 行：训练模式识别（新格式）

```python
        if "pool_idx_2d" in arrays:
            pool_idx_2d = np.asarray(arrays["pool_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(pool_idx_2d.shape[0]),
                "pool_idx_2d": pool_idx_2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }
```

**条件**：存在 `pool_idx_2d`（新format）

**返回**：
```python
{
    "mode": "train_pool",
    "num_samples": N,              # pool行数
    "pool_idx_2d": ndarray[N, M],  # 对象数组或int数组
    "anchor_idx": array[N]         # 锚点（可选）
}
```

**pool_idx_2d说明**：
- shape: `(num_samples, max_pool_width)`
- 值：全局trace索引（≥0）或填充标记（-1）
- 例如：
  ```
  pool_idx_2d = [
      [0, 5, 12, 18, -1],     # 样本0的候选pool
      [3, 7, 9, -1, -1],      # 样本1的候选pool
      [1, 2, 4, 6, 8]         # 样本2的候选pool
  ]
  ```

---

### 第 138-145 行：训练模式识别（中间格式）

```python
        if "patch_idx_2d" in arrays and self.train:
            patch_idx_2d = np.asarray(arrays["patch_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(patch_idx_2d.shape[0]),
                "pool_idx_2d": patch_idx_2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }
```

**条件**：有 `patch_idx_2d` 且 `self.train=True`

**作用**：兼容旧format，别名处理（`patch_idx_2d` → `pool_idx_2d`）

---

### 第 147-152 行：遗留模式识别

```python
        if "patch_idx_2d" in arrays and (not self.train):
            patch_idx_2d = np.asarray(arrays["patch_idx_2d"], dtype=np.int64)
            return {
                "mode": "legacy",
                "num_samples": int(patch_idx_2d.shape[0]),
                "patch_idx_2d": patch_idx_2d,
            }
```

**条件**：有 `patch_idx_2d` 但 `self.train=False`

**作用**：兼容旧推理format（现已弃用）

---

### 第 154-159 行：最后备选方案

```python
        if "0" in arrays:
            patch_idx_2d = np.asarray(arrays["0"], dtype=np.int64)
            return {
                "mode": "legacy",
                "num_samples": int(patch_idx_2d.shape[0]),
                "patch_idx_2d": patch_idx_2d,
            }
```

**兼容最旧格式**：单数组 `.npz` 保存为 `np.save("file", array)`

---

### 第 161-165 行：错误处理

```python
        raise ValueError(
            "Unsupported dataset_neighbors format. Expected legacy ['0'], "
            "train pool keys, or infer query/context keys."
        )
```

**关键错误信息**：列出支持的所有格式

---

### 第 170-173 行：索引提取工具 - `_index_row()`

```python
    def _index_row(self, storage: np.ndarray, idx: int) -> np.ndarray:
        row = np.asarray(storage[idx], dtype=np.int64).reshape(-1)
        return row[row >= 0]
```

**作用**：从对象数组/2D数组中提取一行，并过滤无效索引

**例子**：
```python
storage = np.array([[1, 5, 12, -1], [3, 7, 9, -1]], dtype=object)
idx = 0
row = storage[0]                    # [1, 5, 12, -1]
row = np.asarray(...).reshape(-1)   # [1, 5, 12, -1]
return row[row >= 0]                # [1, 5, 12]
```

**目的**：-1 是padding标记，需要过滤掉

---

### 第 175-189 行：批量索引工具 - `_take_rows()`

```python
    def _take_rows(self, dataset, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            sample = np.asarray(dataset[:1])
            if sample.ndim == 1:
                return np.zeros((0,), dtype=sample.dtype)
            return np.zeros((0, sample.shape[1]), dtype=sample.dtype)
```

**第 175-184 行**：处理空索引

- 如果 `idx` 为空，返回空数组（形状正确）
- 例子：索引为空 → 返回 `(0, T)` 形状数组

```python
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        out = np.asarray(dataset[sorted_idx])
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return out[inv]
```

**第 185-189 行**：乱序恢复算法

**关键优化**：H5 / HDF5 按行顺序读取快，所以：
1. **排序索引**：`sorted_idx = idx[np.argsort(idx)]`
   ```
   idx = [5, 2, 8]
   order = [1, 0, 2]      # argsort结果
   sorted_idx = [2, 5, 8]
   ```

2. **批量读取**：`dataset[sorted_idx]` 一次性读取
   ```
   out = [data[2], data[5], data[8]]
   ```

3. **逆序恢复**：`inv[order] = range(len(order))`
   ```
   inv = [1, 0, 2]
   out[inv] = [data[5], data[2], data[8]]  ← 恢复原始顺序
   ```

**性能**：比逐个读取快得多！

---

### 第 194-202 行：时间裁剪/填充 - `_crop_or_pad_time()`

```python
    def _crop_or_pad_time(self, traces: np.ndarray) -> np.ndarray:
        traces = np.asarray(traces)
        if traces.ndim != 2:
            raise ValueError(f"traces must be 2D [N, T], got {traces.shape}")
        diff = traces.shape[1] - self.time_ps
        if diff > 0:
            return traces[:, diff:]        # 裁剪：取最后 time_ps 个点
        if diff < 0:
            return np.pad(traces, ((0, 0), (-diff, 0)), "constant", constant_values=0)
        return traces
```

**目的**：确保所有traces都有相同的时间长度 `self.time_ps`

**三种情况**：

| 情况 | 长度 | 操作 | 示例 |
|------|------|------|------|
| `diff > 0` | T > 1256 | 裁剪末尾 | 1500 → 1256 (取[:, 244:]) |
| `diff < 0` | T < 1256 | 填充开头 | 1000 → 1256 (pad 256 个0) |
| `diff = 0` | T = 1256 | 无操作 | 保持不变 |

**为什么填充开头？**
- `((0, 0), (-diff, 0))` 表示：行不填，列前填 `-diff`个，后填0个
- 意义：时间轴从前向后，缺失的时间点（比如仪器未启动）应该填充在开头

---

### 第 204-209 行：时间轴生成 - `_time_axis_2d()`

```python
    def _time_axis_2d(self, n_trace: int) -> np.ndarray:
        time_idx_1d = np.arange(0, self.time_ps, dtype=np.int32)
        time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
        return np.tile(time_axis_1d[None, :], (int(n_trace), 1)).astype(np.float32)
```

**步骤**：

1. **第 205 行**：0 ~ 1255 的整数
   ```python
   time_idx_1d = [0, 1, 2, ..., 1255]
   ```

2. **第 206 行**：转换为毫秒
   ```python
   time_axis_1d = 0 + [0,1,2,...,1255] * 4 = [0, 4, 8, ..., 5020] ms
   ```

3. **第 207 行**：复制 `n_trace` 次
   ```python
   result shape: (n_trace, 1256)
   result[0] = result[1] = ... = [0, 4, 8, ..., 5020]
   ```

**用途**：作为网络输入的坐标编码，表达每条trace的时间戳

---

### 第 211-227 行：数据缩放 - `_scale_pair()` 【★★ 关键】

```python
    def _scale_pair(
        self,
        data_patch: np.ndarray,
        masked_patch: np.ndarray,
        is_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
        obs = masked_patch[~is_query]  # ① 提取context部分
        obs = obs[np.isfinite(obs)]    # ② 过滤无穷/NaN
        std_val = np.float32(np.std(obs)) if obs.size > 0 else np.float32(0.0)
        std_val = np.float32(max(std_val, 1e-2))
```

**第 211-217 行**：计算标准差

```
masked_patch = [[0, 2, 3],      (query被掩码为0)
                 [1, 2, 5]]      (context保留观测值)

is_query = [True, False]
obs = masked_patch[~is_query] = [[1, 2, 5]]
std_val = std([1, 2, 5]) ≈ 1.88
```

**第 218 行**：标准差下限（防止除零）
```python
std_val = max(std_val, 0.01)  # ≥ 0.01 ms 级别
```

---

### 第 218-223 行：计算阈值

```python
        ref = np.abs(obs) if obs.size > 0 else np.abs(masked_patch[np.isfinite(masked_patch)])
        thres = np.percentile(ref, 99.5) if ref.size > 0 else 1e-6
        thres = float(max(thres, 1e-6))
        masked_patch = np.clip(masked_patch, -thres, thres) / thres
        data_patch = np.clip(data_patch, -thres, thres) / thres
        self.std_val = std_val
```

**第 218 行**：取context的绝对值
```python
ref = abs(obs)
```

**第 219 行**：99.5 percentile 作为剪裁阈值
```python
thres = np.percentile(ref, 99.5)
# 例如：如果ref=[1, 1, 2, 2, 3, 3, 4, 4, 1000]，
# 则 99.5 percentile ≈ 997（排除极端异常值）
```

**第 220 行**：阈值下限
```python
thres = max(thres, 1e-6)  # 防止过小
```

**第 221-222 行**：归一化
```python
masked_patch = clip(masked_patch, -thres, thres) / thres
data_patch = clip(data_patch, -thres, thres) / thres
# 结果范围：[-1, 1]
```

**第 223 行**：保存标准差用于反归一化
```python
self.std_val = std_val
```

**返回**：四元组 `(data_patch, masked_patch, std_val, thres)`

---

### 第 229-231 行：采样随机生成器 - `_sample_rng()`

```python
    def _sample_rng(self, idx: int) -> np.random.Generator:
        seed = int(self._rng.integers(0, 2**31 - 1)) ^ int(idx)
        return np.random.default_rng(seed)
```

**作用**：为每个样本生成确定的随机数生成器

**例子**：
```python
idx = 0
seed = (某个数) ^ 0
rng = Generator(seed)

idx = 1
seed = (某个数) ^ 1    # 不同的seed
rng = Generator(seed)
```

**好处**：
- 多进程DataLoader中，每个样本的采样是确定的
- 但不同epoch可能看到不同的query/context组合（因为 `self._rng` 会推进）

---

### 第 236-243 行：轨迹排序 - `_sort_traces()` 【★ 推理关键】

```python
    def _sort_traces(
        self,
        data_patch: np.ndarray,
        is_query: np.ndarray,
        coords_patch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.trace_sort_keys:
            order = np.arange(data_patch.shape[0])
            return data_patch, is_query, coords_patch, order
        cols = [coords_patch[:, self._COORD_COL[k]] for k in reversed(self.trace_sort_keys)]
        order = np.lexsort(cols)
        return data_patch[order], is_query[order], coords_patch[order], order
```

**条件1**：无排序键
```python
if not self.trace_sort_keys:
    return 原样不变
```

**条件2**：按指定维度排序
```python
# 假设 trace_sort_keys = ("recv_stake", "recv_line", "shot_stake", "shot_line")
# self._COORD_COL = {"sx": 0, "sy": 1, "rx": 2, "ry": 3, ...}
# 但 recv_stake/recv_line 在H5的其他字段中，需要额外映射

cols = [coords_patch[:, COORD_COL["recv_stake"]],
        coords_patch[:, COORD_COL["recv_line"]],
        coords_patch[:, COORD_COL["shot_stake"]],
        coords_patch[:, COORD_COL["shot_line"]]]

order = np.lexsort(cols)  # 按倒序优先级排序
```

**lexsort说明**：
- `np.lexsort([b, a])` 按 a 升序，a相同时按 b 升序
- 这里倒序列表，最后一个(shot_line)优先级最低

**为什么排序？**
- 地球物理数据通常按shot/receiver线组织
- 排序使相邻traces在空间上接近 → 模型学习更好

---

### 第 248-252 行：坐标归一化 - `_normalize_coords()`

```python
    def _normalize_coords(self, sx, sy, rx, ry) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stats = self.coord_stats
        sx_n = 2 * (sx - stats["sx_min"]) / (stats["sx_max"] - stats["sx_min"]) - 1
        sy_n = 2 * (sy - stats["sy_min"]) / (stats["sy_max"] - stats["sy_min"]) - 1
        rx_n = 2 * (rx - stats["rx_min"]) / (stats["rx_max"] - stats["rx_min"]) - 1
        ry_n = 2 * (ry - stats["ry_min"]) / (stats["ry_max"] - stats["ry_min"]) - 1
        return sx_n, sy_n, rx_n, ry_n
```

**公式**：Min-Max 归一化到 [-1, 1]

```
x_norm = 2 * (x - min) / (max - min) - 1
       = 2 * (x - min) / range - 1
```

**例子**：
```
sx_min=0, sx_max=1000
sx = 250
sx_n = 2 * (250 - 0) / (1000 - 0) - 1 = 2 * 0.25 - 1 = -0.5
```

**结果范围**：`[-1, 1]`（除非数据超出统计范围）

---

### 第 254-290 行：坐标统计计算 - `compute_coord_stats()` 【★★ 关键】

```python
    def compute_coord_stats(self):
        sx_all = self.h5_data_regular["sx"]
        sy_all = self.h5_data_regular["sy"]
        rx_all = self.h5_data_regular["rx"]
        ry_all = self.h5_data_regular["ry"]
```

**第 254-259 行**：读取规则网格坐标

```python
        dsx, sx_u = self.typical_grid_step(sx_all)
        dsy, sy_u = self.typical_grid_step(sy_all)
        drx, rx_u = self.typical_grid_step(rx_all)
        dry, ry_u = self.typical_grid_step(ry_all)
```

**第 260-263 行**：计算网格步长

```python
        sx_min, sx_max = float(sx_u.min()), float(sx_u.max())
        sy_min, sy_max = float(sy_u.min()), float(sy_u.max())
        rx_min, rx_max = float(rx_u.min()), float(rx_u.max())
        ry_min, ry_max = float(ry_u.min()), float(ry_u.max())
```

**第 264-267 行**：提取坐标范围

```python
        deltas = {}
        if dsx is not None and (sx_max - sx_min) > 0:
            deltas["sx"] = float((sx_max - sx_min) / (2 * dsx))
        # ... 类似处理 sy, rx, ry
        self.scale = deltas
```

**第 268-278 行**：计算p_scale因子

**p_scale 概念**（地球物理术语）：
- 不同维度的物理单位/尺度可能差异很大
- p_scale 用来调整每个维度的权重

例如：
- sx 范围：0-10000 m
- sy 范围：0-8000 m  
- rx 范围：0-10000 m
- ry 范围：0-8000 m

则 `deltas["sx"] = (10000 - 0) / (2 * 50) = 100`

---

### 第 279-289 行：构建返回字典

```python
        stats = {
            "sx_min": sx_all.min(),
            "sx_max": sx_all.max(),
            ...
            "grid_step_sx": dsx,
            ...
        }

        if self.use_p_scale and self.scale:
            for name in ("sx", "sy", "rx", "ry"):
                s = self.scale.get(name)
                if s is not None:
                    stats[f"{name}_min"] *= s
                    stats[f"{name}_max"] *= s
```

**if use_p_scale**：缩放坐标范围
```python
stats["sx_min"] *= deltas["sx"]
stats["sx_max"] *= deltas["sx"]
# ... 其他维度
```

---

### 第 290-292 行：计算特征长度

```python
        stats["Lx"] = 0.5 * max(
            stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"]
        )
        stats["Ly"] = 0.5 * max(
            stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"]
        )
        return stats
```

**含义**：
- `Lx` = 0.5 × max(shot方向范围, receiver方向范围)
- 用作坐标的特征尺度，用于distance-aware损失等

---

### 第 297-379 行：训练样本构建 - `_build_train_query_context_sample()` 【★★★ 核心】

这是训练最关键的方法。

```python
    def _build_train_query_context_sample(self, idx: int) -> Dict[str, Any]:
        pool_idx = self._index_row(self.patch_meta["pool_idx_2d"], idx)
```

**第 297-298 行**：提取该样本的候选pool

```python
pool_idx = [1, 5, 12, 18, 23, ...]  # 该样本的全局trace索引
```

```python
        if pool_idx.size < 2:
            raise RuntimeError("train pool must contain at least 2 traces")
```

**第 299 行**：验证pool至少有2条traces（1条query + 1条context）

```python
        anchor_idx = None
        if self.patch_meta.get("anchor_idx") is not None:
            anchor_idx = int(np.asarray(self.patch_meta["anchor_idx"])[idx])
```

**第 301-302 行**：读取该样本的锚点（可选）

---

### 第 304-309 行：加载pool的数据和坐标

```python
        data_pool = self._crop_or_pad_time(
            self._take_rows(self.h5_data["data"], pool_idx)
        ).astype(np.float32)
        rx_pool = self._take_rows(self.h5_data["rx"], pool_idx).astype(np.float32)
        ry_pool = self._take_rows(self.h5_data["ry"], pool_idx).astype(np.float32)
        sx_pool = self._take_rows(self.h5_data["sx"], pool_idx).astype(np.float32)
        sy_pool = self._take_rows(self.h5_data["sy"], pool_idx).astype(np.float32)
```

**作用**：从观测H5读取pool内所有traces的数据和坐标

**数据形状**：
- `data_pool`: `(pool_size, 1256)` 
- `rx_pool, ry_pool, sx_pool, sy_pool`: `(pool_size,)`

---

### 第 311-312 行：归一化坐标

```python
        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(sx_pool, sy_pool, rx_pool, ry_pool)
        coords_pool = np.stack([sx_n, sy_n, rx_n, ry_n], axis=1).astype(np.float32)
```

**作用**：
1. 将每个坐标归一化到 [-1, 1]
2. 堆叠成 `(pool_size, 4)` 数组

---

### 第 314-328 行：随机采样query/context 【关键采样逻辑】

```python
        rng = self._sample_rng(idx)
```

**第 314 行**：获取该样本专属的RNG

```python
        q_eff = min(self.train_num_query, int(pool_idx.size) - 1)
        if q_eff < 1:
            raise RuntimeError("effective train query count must be >= 1")
```

**第 315-317 行**：计算实际query数量
```python
q_eff = min(16, pool_size - 1)  # 至多16，但至少留1条作为context
```

```python
        k_ctx_target = (
            max(1, self.trace_ps - q_eff)
            if self.train_context_size is None
            else self.train_context_size
        )
```

**第 318-321 行**：计算目标context数量
```python
# 如果 train_context_size 未指定（None）
k_ctx_target = max(1, 128 - q_eff)  # 例如：128 - 8 = 120

# 如果指定了
k_ctx_target = train_context_size  # 使用用户指定值
```

```python
        k_ctx = min(int(k_ctx_target), int(pool_idx.size) - q_eff)
        if k_ctx < 1:
            raise RuntimeError("effective train context count must be >= 1")
```

**第 322-324 行**：实际context数量（受pool大小限制）
```python
k_ctx = min(120, pool_size - q_eff)
```

---

### 第 326-343 行：query/context选择 【多样性采样】

```python
        perm = rng.permutation(pool_idx.size)
```

**第 326 行**：随机排列pool中的所有traces

```python
        if (
            self.force_anchor_query
            and anchor_idx is not None
            and np.any(pool_idx == anchor_idx)
        ):
```

**第 327-329 行**：强制锚点约束条件

```python
            anchor_local = int(np.flatnonzero(pool_idx == anchor_idx)[0])
            rest = perm[perm != anchor_local]
            extra = rest[: max(0, q_eff - 1)]
            query_local = np.concatenate(
                [np.asarray([anchor_local], dtype=np.int64), extra.astype(np.int64)],
                axis=0,
            )
```

**第 330-335 行**：如果强制锚点
1. 找到锚点在pool中的局部索引
2. 其他随机位置中选 `q_eff-1` 条
3. 拼接：`[anchor, ...其他]`

```python
        else:
            query_local = perm[:q_eff].astype(np.int64, copy=False)
```

**否则**：直接从随机排列的前 `q_eff` 个

---

### 第 345-356 行：多样性context选择 【核心：diverse_topk】

```python
        candidate_local = np.asarray(
            [i for i in range(pool_idx.size) if i not in set(query_local.tolist())],
            dtype=np.int64,
        )
```

**第 345-349 行**：候选集 = pool - query
```python
query_local = [2, 5, 8]
candidate_local = [0, 1, 3, 4, 6, 7, ...]  # 不在query中的索引
```

```python
        center_coord = np.mean(coords_pool[query_local], axis=0).astype(np.float32, copy=False)
```

**第 351 行**：query的中心坐标
```python
center_coord = mean([coords_pool[2], coords_pool[5], coords_pool[8]])
# shape: (4,)  表示查询点的平均位置
```

```python
        context_local = diverse_topk(
            center_coord=center_coord,
            candidate_idx=candidate_local,
            all_coords=coords_pool,
            k=k_ctx,
            metric_weights=self.patch_metric_weights,
            beta=self.patch_beta,
        ).astype(np.int64, copy=False)
```

**第 353-359 行**：调用 `diverse_topk()`

**diverse_topk 的工作原理**：
```
score(i) = -dist²(center, candidate[i]) + beta * min_dist²(candidate[i], selected)

贪心选择：
  1. 最接近center的候选
  2. 第2个：权衡距离center + 距离已选的多样性
  3. ...
```

**目的**：选出K条既接近query又相互分散的context traces

---

### 第 361-365 行：patch组装

```python
        if context_local.size == 0:
            raise RuntimeError("failed to build non-empty training context from pool")

        patch_local = np.concatenate([query_local, context_local], axis=0)
        data_patch = data_pool[patch_local].astype(np.float32, copy=False)
        is_query_orig = np.zeros((patch_local.size,), dtype=bool)
        is_query_orig[: query_local.size] = True
```

**第 361-363 行**：验证context非空

**第 364-368 行**：组装patch
```python
patch_local = [2, 5, 8, 0, 1, 3, ...]  # query前，context后
data_patch = data_pool[patch_local]     # shape: (Q+K, T)
is_query_orig = [T, T, T, F, F, F, ...] # query标记
```

---

### 第 370-372 行：坐标提取和排序

```python
        coords_patch = coords_pool[patch_local].astype(np.float32, copy=False)
        data_patch, is_query, coords_patch, _ = self._sort_traces(
            data_patch, is_query_orig, coords_patch
        )
```

**第 370 行**：提取patch的坐标

**第 371-372 行**：按shot/recv线排序
```python
# 排序后：
data_patch:     [(Q+K), T]，按shot/recv组织
is_query:       [Q+K]，排序后的query标记
coords_patch:   [(Q+K), 4]，排序后的坐标
```

---

### 第 374-377 行：生成掩码和缩放

```python
        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
```

**第 374-375 行**：创建掩码版本（query为0）

```python
        data_patch, masked_patch, std_val, thres = self._scale_pair(
            data_patch, masked_patch, is_query
        )
```

**第 376-377 行**：归一化到 [-1, 1]

---

### 第 378-398 行：构建返回字典

```python
        return {
            "data": data_patch,                                          # 完整数据
            "masked_patch": masked_patch,                                # query=0
            "rx_patch": coords_patch[:, 2].astype(np.float32, copy=False),  # receiver X
            "ry_patch": coords_patch[:, 3].astype(np.float32, copy=False),  # receiver Y
            "sx_patch": coords_patch[:, 0].astype(np.float32, copy=False),  # shot X
            "sy_patch": coords_patch[:, 1].astype(np.float32, copy=False),  # shot Y
            "time_axis_2d": self._time_axis_2d(patch_local.size),           # 时间轴
            "std_val": std_val,                                             # 标准差
            "is_query": is_query,                                           # 标记
            "query_count": np.int64(query_local.size),                      # 统计
            "context_count": np.int64(context_local.size),
            "query_global_idx": pool_idx[query_local].astype(np.int64, copy=False),  # 全局索引
            "context_global_idx": pool_idx[context_local].astype(np.int64, copy=False),
            "pool_global_idx": pool_idx.astype(np.int64, copy=False),
            "anchor_global_idx": np.int64(-1 if anchor_idx is None else anchor_idx),
            **amplitude_metadata(thres),  # 幅度元数据
        }
```

**关键字段**：
- `data` / `masked_patch`：训练数据（输入/label）
- 坐标：4D位置信息
- 时间轴：时间编码
- 标记：区分query/context
- 全局索引：用于回溯原始数据

---

### 第 402-464 行：推理样本构建 - `_build_infer_query_context_sample()` 【★★★ 核心】

```python
    def _build_infer_query_context_sample(self, idx: int) -> Dict[str, Any]:
        query_idx = self._index_row(self.patch_meta["grid_query_idx_list"], idx)
        context_idx = self._index_row(self.patch_meta["context_idx_list"], idx)
        if query_idx.size == 0 or context_idx.size == 0:
            raise RuntimeError("infer sample must contain non-empty query and context")
```

**第 402-405 行**：读取预计算的query/context索引

```python
query_idx = [100, 101, 102, ...]      # 规则网格中要补插的indices
context_idx = [5, 12, 18, 23, ...]    # 观测中的条件traces
```

---

### 第 407-415 行：加载query数据（来自规则网格）

```python
        query_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data_regular["data"], query_idx)
        ).astype(np.float32)
        rx_q = self._take_rows(self.h5_data_regular["rx"], query_idx).astype(np.float32)
        ry_q = self._take_rows(self.h5_data_regular["ry"], query_idx).astype(np.float32)
        sx_q = self._take_rows(self.h5_data_regular["sx"], query_idx).astype(np.float32)
        sy_q = self._take_rows(self.h5_data_regular["sy"], query_idx).astype(np.float32)
```

**注意**：从 `h5_data_regular`（规则网格H5）读取，**不是** `h5_data`

---

### 第 417-425 行：加载context数据（来自观测）

```python
        context_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data["data"], context_idx)
        ).astype(np.float32)
        rx_c = self._take_rows(self.h5_data["rx"], context_idx).astype(np.float32)
        ry_c = self._take_rows(self.h5_data["ry"], context_idx).astype(np.float32)
        sx_c = self._take_rows(self.h5_data["sx"], context_idx).astype(np.float32)
        sy_c = self._take_rows(self.h5_data["sy"], context_idx).astype(np.float32)
```

**注意**：从 `h5_data`（观测H5）读取

---

### 第 427-431 行：拼接并标记

```python
        data_patch = np.concatenate([query_data, context_data], axis=0).astype(
            np.float32, copy=False
        )
        is_query_orig = np.zeros((data_patch.shape[0],), dtype=bool)
        is_query_orig[: query_idx.size] = True
```

**作用**：
```python
data_patch = [query; context]  # query在前
is_query_orig = [T,T,...T, F,F,...F]
```

---

### 第 433-443 行：坐标归一化

```python
        sx_qn, sy_qn, rx_qn, ry_qn = self._normalize_coords(sx_q, sy_q, rx_q, ry_q)
        sx_cn, sy_cn, rx_cn, ry_cn = self._normalize_coords(sx_c, sy_c, rx_c, ry_c)

        coords_patch = np.stack(
            [
                np.concatenate([sx_qn, sx_cn]),
                np.concatenate([sy_qn, sy_cn]),
                np.concatenate([rx_qn, rx_cn]),
                np.concatenate([ry_qn, ry_cn]),
            ],
            axis=1,
        ).astype(np.float32)
```

**作用**：
1. 分别归一化query和context坐标
2. 拼接：query + context
3. 堆叠成 `(Q+K, 4)` 数组

---

### 第 445-449 行：排序和掩码

```python
        data_patch, is_query, coords_patch, _order = self._sort_traces(
            data_patch, is_query_orig, coords_patch
        )
        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
```

**作用**：按shot/recv排序，然后生成掩码版本

---

### 第 450-453 行：保存原始版本

```python
        data_raw = data_patch.astype(np.float32, copy=True)
        masked_raw = masked_patch.astype(np.float32, copy=True)
        data_patch, masked_patch, std_val, thres = self._scale_pair(
            data_patch, masked_patch, is_query
        )
```

**作用**：保存缩放前的版本（用于metrics计算）

---

### 第 455-466 行：提取shot/recv线信息

```python
        sl_q = self._take_rows(self.h5_data_regular["shot_line"], query_idx)
        ss_q = self._take_rows(self.h5_data_regular["shot_stake"], query_idx)
        rl_q = self._take_rows(self.h5_data_regular["recv_line"], query_idx)
        rs_q = self._take_rows(self.h5_data_regular["recv_stake"], query_idx)
        sl_c = self._take_rows(self.h5_data["shot_line"], context_idx)
        ss_c = self._take_rows(self.h5_data["shot_stake"], context_idx)
        rl_c = self._take_rows(self.h5_data["recv_line"], context_idx)
        rs_c = self._take_rows(self.h5_data["recv_stake"], context_idx)
```

**作用**：提取元数据（用于可视化或后处理）

---

### 第 467-474 行：组装元数据字典

```python
        patch_info = {
            "shot_line": np.concatenate([sl_q, sl_c])[_order],
            "shot_stake": np.concatenate([ss_q, ss_c])[_order],
            "recv_line": np.concatenate([rl_q, rl_c])[_order],
            "recv_stake": np.concatenate([rs_q, rs_c])[_order],
        }
```

**作用**：按排序顺序组织元数据

---

### 第 475-500 行：构建返回字典

```python
        out = {
            "data": data_patch,
            "masked_patch": masked_patch,
            "rx_patch": coords_patch[:, 2].astype(np.float32, copy=False),
            "ry_patch": coords_patch[:, 3].astype(np.float32, copy=False),
            "sx_patch": coords_patch[:, 0].astype(np.float32, copy=False),
            "sy_patch": coords_patch[:, 1].astype(np.float32, copy=False),
            "time_axis_2d": self._time_axis_2d(data_patch.shape[0]),
            "std_val": std_val,
            "is_query": is_query,
            "query_count": np.int64(query_idx.size),
            "context_count": np.int64(context_idx.size),
            "grid_query_idx": query_idx.astype(np.int64, copy=False),
            "context_idx": context_idx.astype(np.int64, copy=False),
            "patch_info": patch_info,
            "data_raw": data_raw,
            "masked_patch_raw": masked_raw,
            **amplitude_metadata(thres),
        }
```

**与训练的区别**：
- 多了 `data_raw` / `masked_patch_raw`（未缩放）
- 有 `patch_info`（元数据）
- `grid_query_idx` 替代 `query_global_idx`

---

### 第 502-510 行：可选字段

```python
        if self.patch_meta.get("block_id") is not None:
            out["block_id"] = np.int64(np.asarray(self.patch_meta["block_id"])[idx])
        if self.patch_meta.get("block_center_grid_idx") is not None:
            out["block_center_grid_idx"] = np.int64(
                np.asarray(self.patch_meta["block_center_grid_idx"])[idx]
            )
        if self.patch_meta.get("anchor_grid_idx_list") is not None:
            out["anchor_grid_idx"] = self._index_row(
                self.patch_meta["anchor_grid_idx_list"], idx
            )
        return out
```

**作用**：添加可选的块级元数据

---

### 第 515-524 行：调度器 - `__getitem__()`

```python
    def __getitem__(self, idx):
        if self.patch_mode == "train_pool":
            return self._build_train_query_context_sample(idx)

        if (not self.train) and self.patch_mode == "infer_query_context":
            return self._build_infer_query_context_sample(idx)

        raise NotImplementedError(
            f"DatasetH5_all_queryctx: unsupported train={self.train!r}, "
            f"patch_mode={self.patch_mode!r}"
        )
```

**决策树**：
```
if patch_mode == "train_pool":
    → _build_train_query_context_sample()
    
elif patch_mode == "infer_query_context" and not train:
    → _build_infer_query_context_sample()
    
else:
    → NotImplementedError
```

**为什么要检查 `train=False`？**
- train_pool 模式可用于训练或推理（灵活）
- 但推理通常用预计算好的块（infer_query_context）
- 防止混乱：只有推理+预计算 或 训练+pool两种组合

---

## 总结表

| 方法 | 用途 | 关键作用 |
|------|------|--------|
| `__init__` | 初始化 | 加载H5、计算coord_stats、识别patch模式 |
| `_load_h5_group` | H5 I/O | 读取group内所有datasets到内存 |
| `_load_patch_metadata` | 元数据解析 | ★★★ 识别patch划分方式（train_pool vs infer） |
| `_index_row` | 索引工具 | 提取1D数据，过滤-1 padding |
| `_take_rows` | 批读工具 | 优化：排序→读→逆序，加速H5读取 |
| `_crop_or_pad_time` | 时间归一 | 确保所有traces长度相同 |
| `_time_axis_2d` | 时间编码 | 生成时间轴坐标 |
| `_scale_pair` | 数据归一化 | 用context计算99.5% percentile，归一化到[-1,1] |
| `_sample_rng` | 采样RNG | 每个样本专属随机生成器 |
| `_sort_traces` | 轨迹排序 | 按shot/recv线lexsort |
| `_normalize_coords` | 坐标归一 | min-max → [-1, 1] |
| `compute_coord_stats` | 统计计算 | ★★ 计算min/max/grid_step/p_scale |
| `_build_train_query_context_sample` | 训练样本 | ★★★ 随机pool采样 + diverse_topk |
| `_build_infer_query_context_sample` | 推理样本 | ★★★ 预计算块读取 |
| `__getitem__` | 调度 | 根据mode选择采样方式 |

