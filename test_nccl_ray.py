"""
Cross-node NCCL/RCCL all-reduce benchmark via Ray.
Tests IB/RDMA network performance.
"""
import os
import ray
import time
import socket


@ray.remote(num_gpus=1)
class NCCLWorker:
    def __init__(self):
        import torch
        self.hostname = socket.gethostname()

    def get_info(self):
        return {
            "hostname": self.hostname,
            "node_id": ray.get_runtime_context().get_node_id(),
        }

    def get_ainic_info(self):
        import subprocess
        result = subprocess.run(["ip", "-4", "addr", "show"], capture_output=True, text=True)
        ainic = []
        current_iface = None
        for line in result.stdout.split("\n"):
            if ": " in line and "mtu" in line:
                parts = line.split(": ")
                if len(parts) >= 2:
                    current_iface = parts[1].split("@")[0]
            if "inet " in line and current_iface:
                ip = line.strip().split()[1].split("/")[0]
                if ip.startswith("10.224."):
                    ainic.append((current_iface, ip))
        return {"hostname": self.hostname, "ainic": ainic}

    def run_allreduce(self, rank, world_size, master_addr, master_port, nccl_ifname):
        import torch
        import torch.distributed as dist

        if nccl_ifname:
            os.environ["NCCL_SOCKET_IFNAME"] = nccl_ifname
        os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ["NCCL_IB_GID_INDEX"] = "1"

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://{}:{}".format(master_addr, master_port),
            rank=rank,
            world_size=world_size,
        )

        results = {}
        for size_mb in [1, 10, 100, 256]:
            numel = size_mb * 1024 * 1024 // 2  # bf16
            tensor = torch.randn(numel, device="cuda:0", dtype=torch.bfloat16)

            for _ in range(5):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()

            n = 20
            start = time.time()
            for _ in range(n):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            avg_ms = (elapsed / n) * 1000
            bw = (size_mb / 1024) / (avg_ms / 1000)
            results[size_mb] = {"avg_ms": round(avg_ms, 2), "bw_gbps": round(bw, 2)}

        dist.destroy_process_group()
        return {
            "hostname": self.hostname,
            "rank": rank,
            "ifname": nccl_ifname,
            "results": results,
        }


def run_benchmark(node_ids, hostnames, master_addr, ifname, port):
    sep = "=" * 50
    print("\n" + sep)
    print("NCCL_SOCKET_IFNAME={}".format(ifname if ifname else "(default)"))
    print(sep)

    workers = []
    for i, nid in enumerate(node_ids):
        w = NCCLWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=nid, soft=False
            )
        ).remote()
        workers.append(w)

    infos = ray.get([w.get_info.remote() for w in workers])
    for info in infos:
        print("  Worker: {}".format(info["hostname"]))

    try:
        futures = [
            w.run_allreduce.remote(i, 2, master_addr, port, ifname)
            for i, w in enumerate(workers)
        ]
        results = ray.get(futures, timeout=600)
        for r in results:
            if r["rank"] == 0:
                print("  Results:")
                for sz, p in sorted(r["results"].items()):
                    print(
                        "    AllReduce {:>4}MB: {:8.2f}ms  BW: {:6.2f}GB/s".format(
                            sz, p["avg_ms"], p["bw_gbps"]
                        )
                    )
    except Exception as e:
        print("  FAILED: {}".format(e))

    for w in workers:
        ray.kill(w)
    time.sleep(3)


def main():
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"]]
    print("Cluster: {} nodes".format(len(nodes)))

    if len(nodes) < 2:
        print("Need at least 2 nodes")
        ray.shutdown()
        return

    node_ids = [n["NodeID"] for n in nodes[:2]]
    hostnames = [n["NodeManagerHostname"] for n in nodes[:2]]
    master_addr = nodes[0]["NodeManagerAddress"]
    print("Testing: {} <-> {}".format(hostnames[0], hostnames[1]))
    print("Master: {}".format(master_addr))

    # First check AINIC interfaces on both nodes
    print("\n--- AINIC interface check ---")
    for i, nid in enumerate(node_ids):
        w = NCCLWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=nid, soft=False
            )
        ).remote()
        info = ray.get(w.get_ainic_info.remote())
        print("  {}: {} AINIC interfaces".format(info["hostname"], len(info["ainic"])))
        for iface, ip in info["ainic"][:3]:
            print("    {} -> {}".format(iface, ip))
        if len(info["ainic"]) > 3:
            print("    ... and {} more".format(len(info["ainic"]) - 3))
        ray.kill(w)
    time.sleep(2)

    configs = [
        ("ens9np0", 29701),
    ]

    for ifname, port in configs:
        run_benchmark(node_ids, hostnames, master_addr, ifname, port)

    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
