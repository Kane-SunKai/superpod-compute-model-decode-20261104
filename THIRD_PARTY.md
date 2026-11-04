# 第三方来源与本项目边界

## LLMCompass

- 官方项目：https://github.com/PrincetonUniversity/LLMCompass/tree/ISCA_AE
- 许可证：BSD-3-Clause；完整条款保存在根目录 `LICENSE.LLMCompass`。
- 原版权：Copyright (c) 2024, Princeton University。
- 本次参考本机已有的源码快照，未修改原项目。该快照没有可用Git提交号，因此没有伪造commit；文件SHA-256见 `superpod_compute/data/provenance.json`。

实际复用/改写的内容：

1. `software_model/matmul.py::simulate_systolic_array_cycle_count` 中的快速利用率公式（0.99、0.98及窄阵列分支）。本项目以新的接口重新组织，并明确记录每次采用的来源。
2. 原 `look_up_table_16_16.csv` 中的阵列周期数据。35,561行去重为1,049个不同键；相同键没有周期冲突。移除了未使用的utilization列，没有改变cycle值。
3. A100的Core数、频率、矩阵阵列、HBM/L2、原SRAM数值、FP16向量吞吐及启动开销配置。

查表未命中时，本项目采用独立的output-stationary折叠波前公式：

```text
folds = ceil(M/array_rows) * ceil(N/array_cols)
cycles = max(1, folds * (K + array_rows + array_cols - 2) - 1)
```

该公式与捆绑数据的全部1,049个键一致，测试中逐条检查。它不调用SCALE-Sim，不是SCALE-Sim的完整实现，也没有声称对任何数据流、内存停顿或真实Tensor Core都等价。未命中结果标为 `own-os-wavefront-estimate`。硬件mac/cycle按来源方法缩放cycle；更换硬件后仍需要验证适用性。

本项目自己实现：请求与硬件契约、算子插件接口、资源/缓冲调度、K驻留策略、候选生成与搜索、波次及尾块处理、映射缓存、CLI、日志/结果输出和测试。没有把LLMCompass整算子预测结果直接嵌入新流水线，也不依赖原目录、PyTorch、pandas或GPU。

## 资源预算与来源参数的区分

原配置的192 KiB是LLMCompass抽象SRAM参数。本项目按已确认方案使用128 KiB shared预算、128 KiB寄存器预算、32 KiB工作寄存器预留。C按FP32的4 B/元素计容量；没有沿用原阵列数据对象的统一2 B累加口径。

原IO配置保留1 µs latency，但原GEMM的块IO公式仅按带宽计费。默认策略因此不对每个面板加1 µs；可以通过 `charge_hbm_latency` 显式切换，并生成不同缓存键。
