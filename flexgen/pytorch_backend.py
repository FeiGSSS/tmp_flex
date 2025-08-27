"""Implement tensor computations with pytorch."""
from itertools import count
import os
import queue
import shutil
import threading
from typing import Union, Tuple, List

import torch
import numpy as np

from flexgen.utils import (
    GB, vector_gather, str_to_dtype, torch_dtype_to_num_bytes,
    torch_dtype_to_np_dtype
    )
                           
from flexgen.numa_tensor import NumaTensor, add_numa_methods, numa_ops

add_numa_methods()

global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global global_cpu_device


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

    def __init__(self,
                 shape: Union[torch.Size, Tuple[int]],
                 dtype: Union[torch.dtype],
                 data: Union[torch.Tensor],
                 device: Union['TorchDevice', 'TorchDisk'],
                 name: str = None):
        if isinstance(data, torch.Tensor):
            if data.device.type == 'cuda':
                assert str(data.device) == device.device
            else:
                assert data.numa_node() == device.device, f"{data.numa_node()} vs {device.device}"

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
    def create_from_torch(cls,
                          data: torch.Tensor,
                          device: Union['TorchDevice', 'TorchDisk'],
                          name = None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device == 'disk':
            self.device.delete(self)
        self.device = self.data = None

    def load_from_torch(self, torch_tensor: torch.Tensor):
        if self.device.device == 'disk':
            with open(self.data, "wb") as fout:
                torch.save(fout, torch_tensor)
        else:
            self.data.copy_(torch_tensor)
            # Make sure copy_ do not change the device of data
            if self.data.device.type == 'cuda':
                assert str(self.data.device) == self.device.device
            else:
                assert self.data.numa_node() == self.device.device

    def load_from_torch_file(self, filename):
        if self.device.device == 'disk':
            shutil.copy(filename, self.data)
        else:
            self.load_from_torch(torch.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        assert dst.device.startswith("cuda"), f"Only support copy to CUDA devices, not {dst.device}"
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


class TorchDevice:
    def __init__(self, device: str):
        assert device in ['cpu', 'cxl', 'cuda'] or device.startswith('cuda:'), f"Invalid device {device}"
        self.device = device

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device == 'cpu':
            global global_cpu_device
            global_cpu_device = self


    def allocate(self, shape, dtype, pin_memory = True, name=None):
        pin_memory = pin_memory if self.device == 'cpu' else False

        if self.device.startswith('cuda'):
            data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory,
                               device=torch.device(self.device))
        else:
            data = NumaTensor.empty(shape, device=self.device, dtype=dtype)
            if pin_memory:
                if self.device == 'cxl': raise ValueError("CXL does not support pin_memory")
                data = data.pin_memory()
        
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, *kwards):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device != 'cpu':
            return  # Only CPU requires this fp32 workspace

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
            k_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=False)
            v_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=False)
            self.attention_compute_workspace.append((k_cache, v_cache))

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None


    def extend_attention_mask(self,
                              attention_mask: TorchTensor,
                              donate: List[bool]):
        bs = attention_mask.shape[0]
        data = attention_mask.data
        assert self.device == attention_mask.device.device, f"device mismatch {self.device} vs {attention_mask.device.device}"
        device_str = self.device

        if device_str.startswith('cuda'):
            new_data = torch.ones((bs, 1), dtype=data.dtype, device=torch.device(device_str))
            data = torch.concat((data, new_data), dim=1)
        else:
            new_data = NumaTensor.ones((bs, 1), dtype=data.dtype, device=device_str)
            data = torch.concat((data, new_data), dim=1)
            data = data.to_numa(device_str)
            
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)


    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        k_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=False)
        v_cache = self.allocate(shape, str_to_dtype[config.torch_dtype], pin_memory=False)
        return k_cache, v_cache


    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device.startswith('cuda'):
            dev = torch.device(self.device)
            cur_mem = torch.cuda.memory_allocated(dev)
            peak_mem = torch.cuda.max_memory_allocated(dev)
        elif self.device in ('cpu', 'cxl'):
            node_id = NumaTensor.DEVICE_MAP[self.device]
            cur_mem, peak_mem = numa_ops.get_node_mem_stats(node_id)
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.device}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.device}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(device={self.device})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))

        self.device = 'disk'

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

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

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape,
                                  dtype=torch_dtype_to_np_dtype[dtype])
        return TorchTensor(shape, dtype, path, self, name=name)

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


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if src.device.device == 'disk':
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device == 'disk':
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device.startswith('cuda') and
          dst.device.device in ['cpu', 'cxl'] and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device in ['cpu', 'cxl'] and
          dst.device.device.startswith('cuda') and
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


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device == 'disk':
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
        data = data.to_numa('cpu')
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf_fp16 = NumaTensor.empty((1 * GB,), dtype=torch.float16, device='cpu').pin_memory()
    cpu_buf_bf16 = NumaTensor.empty((1 * GB,), dtype=torch.bfloat16, device='cpu').pin_memory()

    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)
            
            if src_data.dtype == torch.float16:
                cpu_buf = cpu_buf_fp16
            elif src_data.dtype == torch.bfloat16:
                cpu_buf = cpu_buf_bf16
            else:
                raise NotImplementedError()

            if (src.device.device.startswith('cuda') or
                dst.device.device.startswith('cuda')):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
