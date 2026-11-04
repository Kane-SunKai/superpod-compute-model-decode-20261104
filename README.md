# BW1100 Decode 计算模型 · 2026-11-04 快照

本仓库独立保存当前使用的计算模型，来源提交为
`89c4203846f13a854903cf65beb8b213e59a428d`，Decode 配置版本为
`decode-waves-final-candidate`。核心代码和配置保持来源版本原样。

原项目：<https://github.com/Kane-SunKai/superpod-compute-model>。
本快照采用独立 Git 历史，不替换原项目。`SNAPSHOT.json` 记录来源及逐文件 SHA-256。

## 运行

Python 3.10+；模型预测只依赖 Python 标准库，不需要 GPU 或 PyTorch。
从本仓库根目录执行：

```bash
python tools/predict_decode.py --m 2 --n 9728 --k 6912 --output results/decode.json
python -m unittest discover -s tests -v
```

第一条命令预计输出约 `108.753 us`，分块为 `[16, 128, 32]`。
测试集含 72 项测试。

## 当前模型的含义

| 项目 | 实现 |
|---|---|
| 输入 | 本卡二维连续 FP16 `A[M,K] @ B[K,N]`；M 为 1～32，N/K 为 64 的倍数 |
| 分块选择 | 从 124 条开发运行的粗粒度执行记录中，按 M/N/K 距离选择模板；不是穷举最优分块 |
| 计算 | MMAC 服务模型：16×16×16，8 cycle；每 CU 4 个矩阵单元 |
| 累加 | 每个输出块在所属 CU 中完成全部 K 累加 |
| 输入流水线 | HBM→L2→Shared→寄存器→矩阵单元；双缓冲及资源约束 |
| 批次 | 预取级别为 1 的模板采用单驻留分批；其他模板按资源容量估计多任务驻留 |
| 输出 | 估算 Shared 写入、L2 写入及 HBM 写回服务；Decode 路径以最大服务时间合并此段 |
| 输出信息 | 微秒时延、分块、批次数、容量预算、各层搬运量和服务时间 |

库中的通用 GEMM mapper 也被保留，但它与 `tools/predict_decode.py` 的模板选择路径不同。
普通 `python -m superpod_compute` 使用通用入口；本快照的 Decode 验证应使用上面的专用命令。

## 当前 BW1100 配置

| 参数 | 数值 |
|---|---:|
| CU 数 / 时钟 | 72 / 1.3 GHz |
| 每 CU Shared | 64 KiB |
| 每 CU 寄存器分区 | 4 个 SIMD × 192 KiB |
| L2 容量 | 8 MiB |
| L2 读 / 写 | 3768 / 2990 Byte/cycle |
| HBM 输入有效读服务率 | 1.83 TB/s |
| HBM 基础带宽（输出路径沿用） | 1.72 TB/s |
| Shared 有效读服务率 | 57.5 Byte/cycle/CU |
| Shared 写服务率 | 约 59.352 Byte/cycle/CU |
| 每算子固定服务时间 | 约 3.121 μs |

配置文件：`superpod_compute/data/bw1100_decode_profile.json`。
有效服务率来自开发数据校准，不能与硬件标称峰值混为一谈。
MMAC 路径按 `instruction_issue_cycles=8` 计算；配置中保留的
`mac_per_cell_per_cycle` 是阵列后端字段，不决定本路径的 MMAC 周期。

## 只分析内核 A 的固定分块方式

当前关注内核 A 的 `16×128×32` 映射以及单驻留分批。
默认预测入口仍会自动选择模板，没有在这次备份中被改为强制 A。
固定映射的模型对照可运行：

```python
import json
from superpod_compute.contracts import Hardware, Policy
from superpod_compute.decode import evaluate

with open("superpod_compute/data/bw1100_decode_profile.json", encoding="utf-8") as f:
    profile = json.load(f)
result = evaluate(
    [2, 18432, 6912], [16, 128, 32],
    Hardware(**profile["hardware"]), Policy(**profile["policy"]),
    profile["input_read_Bps"], profile["shared_read_Bpc"],
    {"task_waves": True},
)
print(result["duration_us"])  # 约 143.054 μs，144 块按 72+72 处理
```

这里是固定执行方案后的成本预测，不是自动选择真实库内核。

## 验证状态

`validation/independent_decode_24.csv` 保存最新 24 组独立验证的矩阵尺寸、
实测时延、冻结预测和误差。预测在实测前确定，未使用这批数据调参。

- 实测方式：FP16 `torch.matmul(A, B, out=C)`；Graph 稳态 GPU Event。
- 预热 30 次；Graph 内 64 或 256 次调用；预热回放 6 次。
- 两轮各 25 个计时样本，每轮取 p50，再取两轮均值。
- 同地址、预分配输出；数值正确性检查通过。
- 12/24 组绝对相对误差 ≤10%；平均绝对相对误差 21.61%，最大 57.64%。

这批默认库调用验证并非只固定内核 A。固定内核 A 的对照与默认库调用必须分开解释。

跨批次预取、独立两级 K 面板搜索和 Split-K 成本试算在本次实验中没有形成可替换版本，
没有合入本快照。没有声称模型已经达到所有 Decode 尺寸误差 ≤10%。

## 文件导航

- `superpod_compute/decode.py`：当前 Decode 模板选择与预测。
- `superpod_compute/scheduler.py`：缓冲和各阶段服务的流水线。
- `superpod_compute/backends.py`：MMAC / 阵列等计算后端。
- `superpod_compute/contracts.py`：硬件、策略和请求接口。
- `tools/predict_decode.py`：Decode 命令行入口。
- `tools/benchmark_gemm.py`：保留的通用 GPU benchmark 工具；其默认测量流程不是上述最新 Graph 验证协议。
- `tests/`：72 项回归测试。
- `THIRD_PARTY.md`、`LICENSE.LLMCompass`：第三方来源及许可。

本仓库不包含厂商分发数据库、反汇编、设备二进制、SSH 配置或仅授权本机分析的实验文件。
运行所需的模板先验为开发运行中观察到的粗粒度数值，不含厂商完整内核名称、选择键或适用条件数据库。
