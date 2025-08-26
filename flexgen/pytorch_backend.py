"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    str_to_dtype, np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

from flexgen.libnuma import libnuma
from concurrent.futures import ThreadPoolExecutor
import ctypes


class MemCopy:
    def __init__(self):
        self.parallel_threshold = 64 * 1024 * 1024
        self.num_threads = 8
        self.stats = {
            'total_copies': 0,
            'optimized_copies': 0,
            'total_bytes': [],
            'optimized_bytes': []
        }
    
    def memmove(self, dst_ptr: int, src_ptr: int, size: int):
        self.stats['total_copies'] += 1
        self.stats['total_bytes'].append(size)

        # print(f"MemCopy: Moving {size / (1024**2):.2f} MB")    
        
        if size < self.parallel_threshold:
            ctypes.memmove(ctypes.c_void_p(dst_ptr), 
                          ctypes.c_void_p(src_ptr), 
                          ctypes.c_size_t(size))
            return
        
        self.stats['optimized_copies'] += 1
        self.stats['optimized_bytes'].append(size)

        num_threads = min(self.num_threads, os.cpu_count())
        
        chunk_size = size // num_threads
        
        def copy_chunk(thread_id):
            start_offset = thread_id * chunk_size
            if thread_id == num_threads - 1:
                copy_size = size - start_offset
            else:
                copy_size = chunk_size
            
            dst_chunk = int(dst_ptr) + int(start_offset)
            src_chunk = int(src_ptr) + int(start_offset)
            
            if dst_chunk < 0 or src_chunk < 0:
                raise ValueError(f"Invalid pointer calculation: dst={dst_chunk}, src={src_chunk}")
            
            ctypes.memmove(ctypes.c_void_p(dst_chunk), 
                           ctypes.c_void_p(src_chunk), 
                           ctypes.c_size_t(copy_size))
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(copy_chunk, i) for i in range(num_threads)]
            for future in futures:
                future.result()
    
    def get_stats(self):
        return self.stats.copy()
    
    def print_stats(self):
        stats = self.get_stats()
        opt_ratio = (stats['optimized_copies'] / max(stats['total_copies'], 1)) * 100
        opt_bytes_ratio = (sum(stats['optimized_bytes']) / max(sum(stats['total_bytes']), 1)) * 100
        ave_bytes = sum(stats['total_bytes']) / max(stats['total_copies'], 1)
        ave_opt_bytes = sum(stats['optimized_bytes']) / max(stats['optimized_copies'], 1)

        print(f"Memory Copy Optimization Stats:")
        print(f"  Total copies: {stats['total_copies']}")
        print(f"  Optimized copies: {stats['optimized_copies']} ({opt_ratio:.1f}%)")
        print(f"  Total bytes: {sum(stats['total_bytes']) / (1024**3):.2f} GB")
        print(f"  Optimized bytes: {sum(stats['optimized_bytes']) / (1024**3):.2f} GB ({opt_bytes_ratio:.1f}%)")
        print(f"  Average bytes per copy: {ave_bytes / (1024**2):.2f} MB")
        print(f"  Average bytes per optimized copy: {ave_opt_bytes / (1024**2):.2f} MB")

memcopy = MemCopy()

def print_memory_copy_stats():
    memcopy.print_stats()

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()
    NUMA = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        elif name == "numa":
            return DeviceType.NUMA
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        if self.device.device_type == DeviceType.NUMA:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_torch(self, torch_tensor):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                torch.save(fout, torch_tensor)
        elif self.device.device_type == DeviceType.NUMA:
            ptr, byte_size, shape, dtype = self.data
            assert torch_tensor.flags.c_contiguous, "NUMA tensor must be C-contiguous"
            memcopy.memmove(ptr, torch_tensor.ctypes.data, byte_size)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = global_cpu_device.compressed_device.compress(torch_tensor, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch_tensor)

    def load_from_torch_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_torch(torch.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
            # shape = torch.tensor(tuple(x.stop - x.start for x in src_indices
            # ) + self.shape[len(src_indices):])
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, self.dtype, self.data[2])
        else:
            ret = dst.allocate(shape, self.dtype)
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")
        
class TorchNuma:
    """Manage tensors stored on a NUMA node."""
    def __init__(self, numa_node: int=2):
        self.numa_node = numa_node
        self._metadata = {}  # {key: (ptr_as_int, byte_size, shape, dtype)}
        self._lock = threading.Lock()
        self.device_type: DeviceType = DeviceType.NUMA
        self.dev = None
        
    def allocate(self, shape, dtype, pin_memory=None, name=None) -> TorchTensor:
        """
        Allocate a tensor on a NUMA node.
        """
        name = name or TorchTensor.next_name()
        byte_size = torch.prod(shape) * torch_dtype_to_num_bytes[str_to_dtype[dtype]]
        ptr = libnuma.alloc_onnode(byte_size, self.numa_node)
        if ptr is None:
            raise MemoryError(f"Failed to allocate {byte_size} bytes on NUMA node {self.numa_node}")
        self._metadata[name] = (ptr, byte_size, shape, dtype)
        
        return TorchTensor(shape,
                           dtype, 
                           (ptr, byte_size, shape, dtype),
                           self,
                           name=name)
        

    def delete(self, tensor: TorchTensor) -> None:
        with self._lock:
            if tensor.name not in self._metadata:
                raise ValueError(f"Tensor {tensor.name} not found in NUMA metadata.")
            ptr, byte_size, _, _ = self._metadata.pop(tensor.name)
            libnuma.free(ptr, byte_size)
        
            
    def init_cache_one_gpu_batch(self, config, task, policy) -> TorchTensor:
        """
        Initialize a cache for one batch on disk.
        """
        n_head = config.n_head
        hidden_size = config.input_dim
        prompt_len = task.prompt_len
        gen_len = task.gen_len
        batch_size = policy.gpu_batch_size
        
        shape = (prompt_len + gen_len - 1, batch_size * n_head, hidden_size // n_head)
        # shape = torch.tensor((prompt_len + gen_len - 1, batch_size * n_head, hidden_size // n_head))
        k_cache = self.allocate(shape, config.torch_dtype)
        v_cache = self.allocate(shape, config.torch_dtype)
        return k_cache, v_cache
    
    def mem_stats(self):
        raise NotImplementedError("NUMA memory stats not implemented")
    
    def print_stats(self, output_file=None):
        raise NotImplementedError("NUMA print stats not implemented")

    def __del__(self):
        if not hasattr(self, '_metadata') or not hasattr(self, '_lock'):
            return
        with self._lock: # Ensure thread safety when freeing memory
            keys = list(self._metadata.keys())
            for key in keys:
                ptr, byte_size, _, _ = self._metadata.pop(key)
                libnuma.free(ptr, byte_size)


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                # shape = torch.tensor((max_seq_len, b * n_head, head_dim))
                k_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=False)
                v_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)


    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=pin_memory)
        v_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=pin_memory)
        return k_cache, v_cache


    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, dtype, # TODO
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, str_to_dtype[config.torch_dtype])
        v_cache = self.allocate(shape, str_to_dtype[config.torch_dtype])
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = (shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:])
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, dtype, # TODO
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, str_to_dtype[config.torch_dtype],
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, str_to_dtype[config.torch_dtype],
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        # if force_io_time is not None:
        #     return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.NUMA or 
            dst.device.device_type == DeviceType.NUMA):
        # The tensor is on NUMA, dispatch to copy threads for asynchronous copy
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            if dst.device.device_type == DeviceType.NUMA:
                assert src.device.device_type != DeviceType.NUMA
                src_data = map_to_torch_tensor(src, src_indices)
                size = np.prod(src_data.shape)
                # 1️⃣  from CUDA to NUMA
                if src.device.device_type == DeviceType.CUDA:
                    # Use a pinned cpu buffer as a relay
                    tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                    tmp_cpu_buf.copy_(src_data)
                    ptr, byte_size, shape, dtype = dst.data
                    assert byte_size >= size * src_data.element_size()
                    # assert byte_size == size * src_data.element_size(), f"Expected {size * src_data.element_size()} bytes, \
                    #                                     got {byte_size} bytes. shape={shape}, dtype={dtype} \
                    #                                         src_data.shape={src_data.shape}, size={size} \
                    #                                             src_data.element_size()={src_data.element_size()} \
                    #                                                 dst.data={dst.data}"
                    memcopy.memmove(dst_ptr=ptr,
                                    src_ptr=tmp_cpu_buf.data_ptr(),
                                    size=size * src_data.element_size())
                # 2️⃣ from CPU to NUMA
                else: 
                    assert src_data.contiguous()
                    ptr, byte_size, shape, dtype = dst.data
                    assert byte_size >= size * src_data.element_size()
                    memcopy.memmove(dst_ptr=ptr,
                                    src_ptr=src_data.data_ptr(),
                                    size=size * src_data.element_size())
                
            elif src.device.device_type == DeviceType.NUMA:
                assert dst.device.device_type != DeviceType.NUMA
                dst_data = map_to_torch_tensor(dst, dst_indices)
                # 3️⃣ from NUMA to CUDA
                if dst.device.device_type == DeviceType.CUDA:
                    # Use a pinned cpu buffer as a relay
                    size = np.prod(dst_data.shape)
                    tmp_cpu_buf = cpu_buf[:size].view(dst_data.shape)
                    ptr, byte_size, shape, dtype = src.data
                    assert byte_size >= size * dst_data.element_size()
                    memcopy.memmove(dst_ptr=tmp_cpu_buf.data_ptr(),
                                    src_ptr=ptr,
                                    size=size * dst_data.element_size())
                    dst_data.copy_(tmp_cpu_buf)
                # 4️⃣ from NUMA to CPU
                else:
                    assert dst_data.contiguous()
                    ptr, byte_size, shape, dtype = src.data
                    size = np.prod(dst_data.shape)
                    assert byte_size >= size * dst_data.element_size()
                    memcopy.memmove(dst_ptr=dst_data.data_ptr(),
                                    src_ptr=ptr,
                                    size=size * dst_data.element_size())
            # 5️⃣ from DISK to CUDA/CPU or vice versa
            else:
                src_data = map_to_torch_tensor(src, src_indices)
                dst_data = map_to_torch_tensor(dst, dst_indices)

                if (src.device.device_type == DeviceType.CUDA or
                    dst.device.device_type == DeviceType.CUDA):
                    # Use a pinned cpu buffer as a relay
                    size = np.prod(src_data.shape)
                    tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                    tmp_cpu_buf.copy_(src_data)
                    dst_data.copy_(tmp_cpu_buf)
                else:
                    dst_data.copy_(src_data)

            queue.task_done()
