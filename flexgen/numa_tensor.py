import torch
from torch.utils.cpp_extension import load

# Compile and load the C++ extension
numa_ops = load(
    name='numa_ops',
    sources=['./flexgen/numa_tensor.cpp'],
    extra_cflags=['-O3'],
    extra_ldflags=['-lnuma'],
    verbose=True
)

class NumaTensor:
    """Wrapper for NUMA-aware tensors"""
    
    # Define device mapping
    DEVICE_MAP = {
        'cpu': 0,    # NUMA node 0 (CPU0 DRAM)
        'cxl': 2,      # Alias for CXL node
    }
    
    @classmethod
    def numa_node(cls, numa_id:int) -> str:
        """Get the device name for a NUMA node ID"""
        
        for k, v in cls.DEVICE_MAP.items():
            if v == numa_id:
                return k

        if numa_id == 1:
            return 'cpu:1'
        
        if numa_id == -1:
            return 'cuda'
        
        raise ValueError(f"Unknown NUMA node ID: {numa_id}")

    @classmethod
    def empty(cls, shape, device: str, dtype=torch.float32, **kwargs):
        """Create empty tensor on specified NUMA node"""
        assert device in cls.DEVICE_MAP, f"Invalid device: {device}"
        numa_node = cls.DEVICE_MAP.get(device)
        
        if isinstance(shape, torch.Size):
            shape = list(shape)
        
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        ret = numa_ops.numa_empty(list(shape), numa_node, dtype)

        return ret

    @classmethod
    def zeros(cls, shape, device: str = 'cpu', dtype=torch.float32, **kwargs):
        """Create zero tensor on specified NUMA node"""
        tensor = cls.empty(shape, device=device, dtype=dtype, **kwargs)
        tensor.zero_()
        return tensor
    
    @classmethod
    def ones(cls, shape, device: str = 'cpu', dtype=torch.float32, **kwargs):
        """Create ones tensor on specified NUMA node"""
        tensor = cls.empty(shape, device=device, dtype=dtype, **kwargs)
        tensor.fill_(1)
        return tensor
    
    @classmethod
    def from_tensor(cls, tensor, device: str = 'cpu'):
        """Convert existing tensor to NUMA tensor on specified node"""
        assert device in cls.DEVICE_MAP, f"Invalid device: {device}"
        numa_node = cls.DEVICE_MAP.get(device)
        return numa_ops.to_numa(tensor, numa_node)
    
    @classmethod
    def get_device(cls, tensor):
        """Get the NUMA node of a tensor"""
        numa_node = numa_ops.get_numa_node(tensor)
        # Reverse lookup in device map
        for device_name, node in cls.DEVICE_MAP.items():
            if node == numa_node:
                return device_name
        return f'cpu:{numa_node}'
    
    
# Monkey-patch torch to add numa methods (optional)
def add_numa_methods():
    """Add NUMA methods to torch.Tensor"""
    
    def to_numa(self, device):
        """Move tensor to NUMA node"""
        return NumaTensor.from_tensor(self, device)
    
    def numa_node(self):
        """Get NUMA node of tensor"""
        # return numa_ops.get_numa_node(self)
        return NumaTensor.numa_node(numa_ops.get_numa_node(self))
    
    torch.Tensor.to_numa = to_numa
    torch.Tensor.numa_node = numa_node