
import ctypes
import ctypes.util

class LibNuma:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LibNuma, cls).__new__(cls)
            cls._instance._init_libnuma()
        return cls._instance

    def _init_libnuma(self):
        """
        Loads the libnuma shared library and defines function prototypes.
        """
        try:
            # Find the library using a robust method
            lib_path = ctypes.util.find_library('numa')
            if not lib_path:
                raise FileNotFoundError("libnuma.so not found. Please install it (e.g., sudo apt-get install libnuma-dev).")
            
            self.lib = ctypes.CDLL(lib_path)
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(f"Failed to load libnuma: {e}")

        # Define function argument types (argtypes) and return types (restype)
        # This is crucial for correctness, especially with pointers and sizes.

        # 1. int numa_available(void);
        self.available = self.lib.numa_available
        self.available.restype = ctypes.c_int

        # 2. void *numa_alloc_onnode(size_t size, int node);
        self.alloc_onnode = self.lib.numa_alloc_onnode
        self.alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
        self.alloc_onnode.restype = ctypes.c_void_p

        # 3. void numa_free(void *start, size_t size);
        self.free = self.lib.numa_free
        self.free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.free.restype = None

        # 4. int numa_max_node(void);
        self.max_node = self.lib.numa_max_node
        self.max_node.restype = ctypes.c_int

# Create a singleton instance for easy access
libnuma = LibNuma()