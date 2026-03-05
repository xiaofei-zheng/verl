# Running verl RL Training with RayJob

Submit verl reinforcement learning training tasks via RayJob on the SaFE platform. The platform automatically creates the Ray cluster and submits the training job.

## Prerequisites

- A ready-to-use training script (e.g., `run_multinode_grpo_fsdp2_no_reshard_offload.sh`)
- Training scripts and datasets stored on shared storage (e.g., `/shared_nfs/`) accessible from all nodes
- A container image with the required Python environment and dependencies

## Step 1: Basic Information

Create a RayJob on the SaFE platform and fill in the **entrypoint** — the command to be submitted to the Ray cluster:

```bash
bash /shared_nfs/xiaofei/verl/rl-scripts/run_multinode_grpo_fsdp2_no_reshard_offload.sh
```

This entrypoint will be submitted to the head node as a Driver process via `ray job submit` once the Ray cluster is ready.

## Step 2: Configure RayCluster

### Image

Both head and worker can use the same image. The recommended pre-verified verl image:

```
harbor.oci-slc.primus-safe.amd.com/custom/lmsysorg/sglang:202603021059
```

You can also use a custom-built image, as long as it includes the dependencies required by verl.

### Resource Configuration

Customize the CPU, memory, and GPU count for head and worker based on your training needs. For example, a 2-node x 8-GPU setup requires 1 head (8 GPUs) + 1 worker (8 GPUs).

### Container Entrypoint (Optional)

Both head and worker configurations have a **container entrypoint**, which differs from the RayJob entrypoint:

| Field | When it runs | Purpose | Required |
|---|---|---|---|
| **RayJob entrypoint** | After Ray cluster is ready | Submit training job to the cluster | Yes |
| **Container entrypoint** | Before `ray start` | Install dependencies, pre-processing | No |

Container entrypoint is useful for:

- Installing dependencies missing from the image (e.g., `pip install xxx`)
- Setting node-level environment variables or configurations
- Running data preparation steps

If the image already contains all dependencies, this can be left empty.

## Step 3: Advanced Options

### Environment Variables

Configure the following environment variable to enable RDMA network acceleration:

| Environment Variable | Value | Description |
|---|---|---|
| `AINIC_DRIVER_VERSION` | `1.117.5-a-56` | Automatically installs the ainic driver and enables RDMA |

This environment variable is configured at the RayCluster pod level and takes effect on all nodes (head + workers).

## Quick Start

You can clone an existing verified job to get started quickly:

**Example Job**: [verl-sglang-fsdp2-qwen3-8b](https://oci-slc.primus-safe.amd.com/rayjob/detail?id=verl-sgalng-fsdp2-qwen3-8b-7cxhw)

Click "Clone" on the job detail page to create a new RayJob based on the existing configuration.

## RayJob Lifecycle

@rayjob-verl-guide-en.md

### Phase Descriptions

1. **Create RayCluster**: KubeRay Operator creates head and worker pods based on the `rayClusterSpec` in the RayJob
2. **Wait for cluster ready**: All pods start up and the Ray cluster initializes successfully
3. **Spawn Submitter Pod**: Once the cluster is ready, the Operator creates a K8s Job to run `ray job submit --address=http://<head>:8265 -- <entrypoint>`
4. **Driver process execution**: The entrypoint script runs on the head node as the Ray Driver
5. **verl connects to cluster**: The training script calls `ray.init()` to connect to the Ray cluster, then creates training workers across nodes via `ray.remote`
6. **Training complete / Cleanup**: After training finishes, the job status is updated and the cluster is cleaned up based on configuration

## Viewing Job Logs

### Real-time Logs During Training

On the SaFE platform's RayJob detail page, navigate to the **Pods** tab to view the status of all pods:

- **Submitter Pod**: A `Succeeded` status indicates the job has been successfully submitted to the Ray cluster
- **Head / Worker Pods**: Click on the corresponding pod and view real-time training logs in the **Logs** section

### Logs After Training Completes

After training completes, go to the **Workload** section and select the corresponding node under **Logs** to view the full log output for that node.

## Important Notes

### Environment Variable Propagation

Environment variables set via `export` in the training script **only take effect in the Driver process on the head node** and are NOT automatically propagated to worker nodes. To set environment variables on all nodes (e.g., NCCL-related configs), use one of the following approaches:

- **Platform environment variables**: Configure in the RayJob advanced options — these are injected into all pods
- **Container Entrypoint**: Export variables in both head and worker container entrypoints

verl automatically propagates certain essential environment variables through Ray's `runtime_env` mechanism (e.g., `TOKENIZERS_PARALLELISM`, `NCCL_CUMEM_ENABLE`, etc.).

### Shared Storage

Ensure that training scripts, model files, and datasets are stored on shared storage paths accessible from all nodes (e.g., `/shared_nfs/`). Otherwise, worker nodes will not be able to read them.
