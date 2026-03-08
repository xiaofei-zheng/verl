# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import os
import queue
import threading
from functools import partial
from time import time

import megatron.core
import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from packaging import version

_rocm_ckpt_patched = False


def _patch_filesystem_writer_for_rocm():
    """Replace fork-based checkpoint writing with thread-based for ROCm.

    ROCm's HIP runtime does not support fork() after GPU context is initialized,
    causing SIGSEGV in Megatron's FileSystemWriterAsync which uses mp.fork.
    Threads share the same address space without forking, avoiding this issue
    while maintaining parallel I/O performance.
    """
    global _rocm_ckpt_patched
    if _rocm_ckpt_patched:
        return
    _rocm_ckpt_patched = True

    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    logger = logging.getLogger(__name__)

    orig_preload = FileSystemWriterAsync.preload_tensors

    @staticmethod
    def _preload_no_pinned(write_buckets, non_blocking=True):
        return orig_preload(write_buckets, non_blocking=False)

    FileSystemWriterAsync.preload_tensors = _preload_no_pinned

    @staticmethod
    def _write_threaded(transform_list, use_msc, rank, write_buckets, global_results_queue):
        """Thread-based replacement for fork-based write_preloaded_data_multiproc."""
        logger_inner = logging.getLogger(__name__)
        w_start = time()
        write_results_or_exc = dict()
        local_results_queue = queue.Queue()
        count_queue = queue.Queue()
        t_list = []

        for i, write_bucket in enumerate(write_buckets):
            try:
                count_queue.put(i)
                kwargs = {
                    "local_proc_idx": i,
                    "write_bucket": write_bucket,
                    "results_queue": local_results_queue,
                    "count_queue": count_queue,
                    "use_fsync": True,
                }
                if use_msc:
                    import inspect
                    signature = inspect.signature(FileSystemWriterAsync.write_preloaded_data)
                    if len(signature.parameters) > 6:
                        kwargs["use_msc"] = use_msc

                t = threading.Thread(
                    target=partial(FileSystemWriterAsync.write_preloaded_data, transform_list),
                    kwargs=kwargs,
                    daemon=True,
                )
                t_list.append(t)
            except Exception as e:
                err_msg = f"Error creating write thread {i}: {e}"
                logger_inner.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception):
            gc.disable()
            try:
                for t in t_list:
                    t.start()

                count_queue.join()

                for proc_idx in range(len(write_buckets)):
                    try:
                        local_proc_idx, local_results_or_exc = local_results_queue.get(timeout=300)
                    except queue.Empty:
                        write_results_or_exc = RuntimeError(
                            f"Timeout waiting for results ({proc_idx}/{len(write_buckets)})"
                        )
                        break
                    else:
                        if isinstance(local_results_or_exc, Exception):
                            logger_inner.error(f"Write thread {local_proc_idx} error: {local_results_or_exc}")
                            write_results_or_exc = local_results_or_exc
                            break
                        assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
                        write_results_or_exc[local_proc_idx] = local_results_or_exc
                        t_list[local_proc_idx].join()
            finally:
                gc.enable()

        global_results_queue.put(write_results_or_exc)
        w_end = time()
        logger_inner.warning(
            f"[ROCm] rank {rank}: threaded checkpoint write took {w_end - w_start:.1f}s "
            f"({len(write_buckets)} buckets)"
        )

    FileSystemWriterAsync.write_preloaded_data_multiproc = _write_threaded
    logger.warning("[ROCm Checkpoint Patch] Replaced fork-based writer with thread-based writer")


def save_dist_checkpointing(
    sharded_state_dict,
    ckpt_path,
    async_save=False,
    content_metadata=None,
):
    _patch_filesystem_writer_for_rocm()
    validate_sharding_integrity = True
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    save_kwargs = dict(
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if content_metadata is not None:
        if mcore_ge_014:
            save_kwargs["content_metadata"] = content_metadata

    return dist_checkpointing.save(sharded_state_dict, ckpt_path, **save_kwargs)


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
