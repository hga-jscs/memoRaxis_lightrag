下面这段可以**直接放进 README**，客户拿到后基本就能照着用了。😊

````md
## 一键运行参数说明（`run_all_tasks.py`）

`run_all_tasks.py` 用于将四个任务串成一条完整链路，按顺序执行：

1. 数据预处理
2. ingest（建索引）
3. infer（推理）
4. evaluate（评测）

默认情况下，它会运行以下四个任务：

- `Accurate_Retrieval`
- `Conflict_Resolution`
- `Long_Range_Understanding`
- `Test_Time_Learning`

最简命令如下：

```bash
python run_all_tasks.py --reset
````

这条命令的含义是：
在运行前重置各任务对应的 LightRAG workspace，然后按默认参数完整执行四个任务。

---

## 1. 通用参数

### `--tasks`

用于指定要运行的任务集合。默认值等价于：

```bash
--tasks acc conflict long ttl
```

可选值如下：

* `acc`：Accurate_Retrieval
* `conflict`：Conflict_Resolution
* `long`：Long_Range_Understanding
* `ttl`：Test_Time_Learning

#### 示例

只运行 Accurate Retrieval 和 Conflict Resolution：

```bash
python run_all_tasks.py --tasks acc conflict --reset
```

只运行 Long Range：

```bash
python run_all_tasks.py --tasks long --reset
```

---

### `--adaptors`

用于指定推理适配器。默认值等价于：

```bash
--adaptors R1 R2 R3
```

可选值：

* `R1`
* `R2`
* `R3`
* `all`

#### 示例

只运行 R1：

```bash
python run_all_tasks.py --adaptors R1 --reset
```

只运行 R1 和 R2：

```bash
python run_all_tasks.py --adaptors R1 R2 --reset
```

---

### `--mode`

用于指定 LightRAG 的检索模式。默认值：

```bash
--mode naive
```

可选值：

* `naive`
* `mix`
* `local`
* `global`
* `hybrid`

#### 示例

使用 `hybrid` 模式运行：

```bash
python run_all_tasks.py --mode hybrid --reset
```

---

### `--storage_dir`

用于指定 LightRAG 索引存储目录。默认值：

```bash
--storage_dir out/lightrag_storage
```

#### 示例

将索引写入另一个目录：

```bash
python run_all_tasks.py --storage_dir out/lightrag_storage_exp1 --reset
```

适用于：

* 区分不同实验
* 避免覆盖已有索引
* 多组对比实验并行保留结果

---

### `--output_suffix`

用于给结果文件名附加后缀，便于区分不同实验。默认值为空：

```bash
--output_suffix ""
```

#### 示例

```bash
python run_all_tasks.py --output_suffix hybrid_r1r2 --adaptors R1 R2 --mode hybrid --reset
```

说明：

* 适合标记不同实验设置
* 例如不同 mode、不同 adaptor 组合
* 但 `Conflict_Resolution` 的官方评测脚本对文件名匹配较敏感，因此默认建议留空

---

### `--reset`

用于在 ingest 前清空对应任务的 workspace。

#### 示例

```bash
python run_all_tasks.py --reset
```

适用于：

* 第一次运行
* 修改了索引构建参数后重建索引
* 怀疑旧索引残留影响结果时

---

### `--skip_preprocess`

跳过 preview JSON 的生成。

#### 示例

```bash
python run_all_tasks.py --skip_preprocess --reset
```

适用于：

* `MemoryAgentBench/preview_samples/` 已经存在
* 不想重复执行数据预处理

---

### `--skip_ingest`

跳过索引构建，直接复用已有索引。

#### 示例

```bash
python run_all_tasks.py --skip_ingest
```

适用于：

* 索引已经构建完成
* 只想修改 adaptor、limit 或评测参数
* 想节省时间

---

### `--skip_infer`

跳过推理阶段。

#### 示例

```bash
python run_all_tasks.py --skip_infer --reset
```

适用于：

* 只想测试 ingest 是否成功
* 只想先构建索引

---

### `--skip_eval`

跳过评测阶段。

#### 示例

```bash
python run_all_tasks.py --skip_eval --reset
```

适用于：

* 只想先生成推理结果
* 暂时不关心评测分数
* 想避开较慢的评测流程

---

## 2. 四个任务的专属参数

---

### Accurate Retrieval

#### `--acc_instance_idx`

指定实例编号。默认值：

```bash
0
```

支持格式：

* 单个编号：`0`
* 区间：`0-3`
* 列表：`0,2,5`
* 混合：`0-2,5,7`

#### 示例

```bash
python run_all_tasks.py --acc_instance_idx 0-5 --reset
```

---

#### `--acc_limit`

限制每个实例回答的问题数。默认值：

```bash
5
```

#### 示例

```bash
python run_all_tasks.py --acc_limit 3 --reset
```

适用于：

* 冒烟测试时减少运行量
* 小规模快速检查结果

---

#### `--acc_chunk_size`

控制 Accurate Retrieval 在 ingest 时的 chunk 大小。默认值：

```bash
850
```

#### 示例

```bash
python run_all_tasks.py --acc_chunk_size 1000 --reset
```

说明：

* 值小：切分更细，可能更利于召回
* 值大：上下文更完整，但可能带入更多噪声

---

### Conflict Resolution

#### `--conflict_instance_idx`

默认值：

```bash
0-7
```

#### 示例

```bash
python run_all_tasks.py --conflict_instance_idx 0-3 --reset
```

---

#### `--conflict_limit`

默认值：

```bash
-1
```

说明：`-1` 表示不限制，全部问题都跑。

#### 示例

```bash
python run_all_tasks.py --conflict_limit 5 --reset
```

---

#### `--conflict_min_chars`

控制 ingest 时最小文本块长度。默认值：

```bash
800
```

#### 示例

```bash
python run_all_tasks.py --conflict_min_chars 1000 --reset
```

说明：

* 值小：文本块更细
* 值大：文本块更完整

---

### Long Range Understanding

#### `--long_instance_idx`

默认值：

```bash
0-39
```

#### 示例

```bash
python run_all_tasks.py --long_instance_idx 0-9 --reset
```

---

#### `--long_limit`

默认值：

```bash
-1
```

说明：`-1` 表示全部问题都跑。

#### 示例

```bash
python run_all_tasks.py --long_limit 2 --reset
```

---

#### `--long_chunk_size`

默认值：

```bash
1200
```

#### 示例

```bash
python run_all_tasks.py --long_chunk_size 1500 --reset
```

---

#### `--long_overlap`

默认值：

```bash
100
```

#### 示例

```bash
python run_all_tasks.py --long_overlap 150 --reset
```

说明：

Long Range 任务对长文本切分更敏感。
若发现长距离推理效果不稳定，可以尝试：

```bash
python run_all_tasks.py --long_chunk_size 1500 --long_overlap 150 --reset
```

---

### Test Time Learning

#### `--ttl_instance_idx`

默认值：

```bash
0-5
```

#### 示例

```bash
python run_all_tasks.py --ttl_instance_idx 0-2 --reset
```

---

#### `--ttl_limit`

默认值：

```bash
-1
```

说明：`-1` 表示全部问题都跑。

#### 示例

```bash
python run_all_tasks.py --ttl_limit 3 --reset
```

---

## 3. 最常见的完整样例

### 样例 1：最简完整运行

四个任务全跑，并在 ingest 前重置 workspace：

```bash
python run_all_tasks.py --reset
```

---

### 样例 2：最小冒烟测试

只跑每个任务的第一个实例，并限制问题数，适合先验证环境是否正常：

```bash
python run_all_tasks.py \
  --acc_instance_idx 0 \
  --conflict_instance_idx 0 \
  --long_instance_idx 0 \
  --ttl_instance_idx 0 \
  --acc_limit 3 \
  --conflict_limit 3 \
  --long_limit 1 \
  --ttl_limit 3 \
  --reset
```

Windows 单行写法：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3 --reset
```

---

### 样例 3：只跑两个任务

只运行 Accurate Retrieval 和 Conflict Resolution：

```bash
python run_all_tasks.py --tasks acc conflict --reset
```

---

### 样例 4：只跑 R1 和 R2

```bash
python run_all_tasks.py --adaptors R1 R2 --reset
```

---

### 样例 5：复用已有索引

跳过 ingest，直接做推理和评测：

```bash
python run_all_tasks.py --skip_ingest
```

---

### 样例 6：只看模型输出，不做评测

```bash
python run_all_tasks.py --skip_eval --reset
```

---

### 样例 7：做一组区分实验结果的运行

```bash
python run_all_tasks.py --adaptors R1 R2 --mode hybrid --output_suffix hybrid_r1r2 --reset
```

---

## 4. 推荐使用顺序

第一次使用仓库时，比较合适的顺序如下：

先做最小冒烟测试：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3 --reset
```

确认环境、路径、接口和数据都正常后，再做标准完整运行：

```bash
python run_all_tasks.py --reset
```

当进入正式实验和调参阶段后，再根据需要改用：

* `--tasks`
* `--adaptors`
* `--mode`
* `--skip_ingest`
* `--output_suffix`

来进行更细粒度的实验控制。

```
```
