# 使用 RayJob 运行 verl RL 训练

在 SaFE 平台上通过 RayJob 提交 verl 的强化学习训练任务，平台会自动创建 Ray 集群并提交训练任务。

## 前提条件

- 已有可用的训练脚本（如 `run_multinode_grpo_fsdp2_no_reshard_offload.sh`）
- 训练脚本和数据存放在共享存储（如 `/shared_nfs/`）上，确保所有节点可访问
- 可用的容器镜像，包含所需的 Python 环境和依赖

## 步骤一：Basic Information

在 SaFE 平台创建 RayJob，填写 **entrypoint**，即提交到 Ray 集群的任务命令：

```bash
bash /shared_nfs/xiaofei/verl/rl-scripts/run_multinode_grpo_fsdp2_no_reshard_offload.sh
```

这个 entrypoint 会在 Ray 集群就绪后，通过 `ray job submit` 提交到 head node 上作为 Driver 进程执行。

## 步骤二：配置 RayCluster

### 镜像

head 和 worker 可以使用相同的镜像，推荐使用已验证的 verl 镜像：

```
harbor.oci-slc.primus-safe.amd.com/custom/lmsysorg/sglang:202603021059
```

也可以使用自己构建的自定义镜像，只需确保镜像中包含 verl 所需的依赖。

### 资源配置

根据训练需求自定义 head 和 worker 的 CPU、内存、GPU 数量。例如 2 节点 x 8 GPU 的配置，需要 1 个 head（8 GPU）+ 1 个 worker（8 GPU）。

### Container Entrypoint（可选）

head 和 worker 的配置中各有一个 **container entrypoint**，它与 RayJob 的 entrypoint 不同：

| 字段 | 执行时机 | 作用 | 是否必填 |
|---|---|---|---|
| **RayJob entrypoint** | Ray 集群就绪后 | 提交训练任务到集群 | 是 |
| **Container entrypoint** | `ray start` 之前 | 安装依赖、前置处理 | 否 |

Container entrypoint 适用于以下场景：

- 安装镜像中缺少的固定依赖（如 `pip install xxx`）
- 设置节点级别的环境变量或配置
- 执行前置数据准备

如果镜像中已包含所有依赖，可以不填写。

## 步骤三：高级选项

### 环境变量配置

需要配置以下环境变量以启用 RDMA 网络加速：

| 环境变量 | 值 | 说明 |
|---|---|---|
| `AINIC_DRIVER_VERSION` | `1.117.5-a-56` | 配置后会自动安装 ainic 驱动，启用 RDMA |

该环境变量配置在 RayCluster 的 pod 级别，会在所有节点（head + worker）上生效。

## 快速体验

可以直接 clone 已跑通的示例任务来快速体验：

**示例任务**：[verl-sglang-fsdp2-qwen3-8b](https://oci-slc.primus-safe.amd.com/rayjob/detail?id=verl-sgalng-fsdp2-qwen3-8b-7cxhw)

在任务详情页点击 clone，即可基于已有配置创建新的 RayJob。

## RayJob 生命周期

@rayjob-lifecycle.drawio

### 各阶段说明

1. **创建 RayCluster**：KubeRay Operator 根据 RayJob 中的 `rayClusterSpec` 创建 head 和 worker pods
2. **等待集群就绪**：所有 pod 启动完成，Ray 集群初始化成功
3. **生成 Submitter Pod**：集群就绪后，Operator 创建一个 K8s Job 来执行 `ray job submit --address=http://<head>:8265 -- <entrypoint>`
4. **Driver 进程执行**：entrypoint 脚本在 head node 上作为 Ray Driver 运行
5. **verl 连接集群**：训练脚本中 `verl` 调用 `ray.init()` 连接到 Ray 集群，然后通过 `ray.remote` 在各节点创建训练 workers
6. **训练完成/清理**：训练结束后，Job 状态更新，根据配置决定是否自动清理集群

## 查看任务日志

### 训练中实时查看

在 SaFE 平台的 RayJob 详情页中，进入 **Pods** 页面可以看到所有 pod 的状态：

- **Submitter Pod**：状态为 `Succeeded` 表示任务已成功提交到 Ray 集群
- **Head / Worker Pods**：点击对应 pod，在 **Logs** 中可以实时查看训练日志

### 训练完成后查看

训练完成后，在 **Workload** 的 **Logs** 中选择对应的节点即可查看该节点的完整日志输出。

## 注意事项

### 环境变量传播

训练脚本中通过 `export` 设置的环境变量**只在 head node 的 Driver 进程中生效**，不会自动传播到 worker 节点。如果需要在所有节点上设置环境变量（如 NCCL 相关配置），有两种方式：

- **平台环境变量**：在 RayJob 的高级选项中配置，会注入到所有 pod
- **Container Entrypoint**：在 head 和 worker 的 container entrypoint 中 export

verl 会通过 Ray 的 `runtime_env` 机制自动传播部分必要的环境变量（如 `TOKENIZERS_PARALLELISM`、`NCCL_CUMEM_ENABLE` 等）。

### 共享存储

确保训练脚本、模型文件、数据集都存放在所有节点可访问的共享存储路径上（如 `/shared_nfs/`），否则 worker 节点将无法读取。
