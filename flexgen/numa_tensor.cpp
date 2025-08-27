// filepath: /home/fei/research/llm/FlexLLMGen/flexgen/numa_tensor.cpp
#include <torch/extension.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <numa.h>
#include <numaif.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <atomic>
#include <tuple>

// 每节点统计结构（用指针存，避免原子不可拷贝问题）
struct NodeMemStat {
    std::atomic<size_t> current{0};
    std::atomic<size_t> peak{0};
};
static inline void update_peak(NodeMemStat& stat, size_t new_cur) {
    size_t prev = stat.peak.load(std::memory_order_relaxed);
    while (new_cur > prev &&
           !stat.peak.compare_exchange_weak(prev, new_cur,
                                            std::memory_order_relaxed)) {}
}

// 先定义上下文
struct NUMAContext {
    void*  ptr;
    size_t size;
    int    node_id;
    NUMAContext(void* p, size_t s, int n): ptr(p), size(s), node_id(n) {}
};

// 先前向声明内存统计 map（用 unique_ptr）
static std::unordered_map<int, std::unique_ptr<NodeMemStat>> node_mem_stats;

// 先声明释放函数，供分配器使用
void numa_deleter(void* ctx);

// 分配器类（需在 allocators 声明前定义，因 make_shared 需要完整类型）
class NUMAAllocator : public c10::Allocator {
private:
    int numa_node_;
public:
    explicit NUMAAllocator(int numa_node) : numa_node_(numa_node) {}
    c10::DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) nbytes = 1;
        void* ptr = numa_alloc_onnode(nbytes, numa_node_);
        if (!ptr) {
            throw std::runtime_error(
                "Failed to allocate on NUMA node " + std::to_string(numa_node_));
        }
        auto* ctx = new NUMAContext(ptr, nbytes, numa_node_);

        // 更新统计
        auto it = node_mem_stats.find(numa_node_);
        if (it != node_mem_stats.end()) {
            size_t new_cur =
                it->second->current.fetch_add(nbytes, std::memory_order_relaxed) + nbytes;
            update_peak(*it->second, new_cur);
        }
        return c10::DataPtr(ptr, ctx, numa_deleter, c10::Device(c10::DeviceType::CPU));
    }
    void copy_data(void* dest, const void* src, size_t count) const override {
        std::memcpy(dest, src, count);
    }
    c10::DeleterFnPtr raw_deleter() const override { return nullptr; }
};

// 现在可以声明分配器表
static std::unordered_map<int, std::shared_ptr<NUMAAllocator>> allocators;

// 释放函数实现
void numa_deleter(void* ctx) {
    NUMAContext* context = static_cast<NUMAContext*>(ctx);
    if (!context) return;
    numa_free(context->ptr, context->size);
    auto it = node_mem_stats.find(context->node_id);
    if (it != node_mem_stats.end()) {
        it->second->current.fetch_sub(context->size, std::memory_order_relaxed);
    }
    delete context;
}

// 初始化
void init_numa_allocators() {
    if (numa_available() < 0) {
        throw std::runtime_error("NUMA is not available on this system");
    }
    if (!allocators.empty()) return;

    int num_nodes = numa_max_node() + 1;
    for (int i = 0; i < num_nodes; ++i) {
        struct bitmask* mask = numa_allocate_cpumask();
        if (numa_node_to_cpus(i, mask) < 0) {
            numa_free_cpumask(mask);
            continue;
        }
        numa_free_cpumask(mask);
        allocators[i] = std::make_shared<NUMAAllocator>(i);
        node_mem_stats.emplace(i, std::make_unique<NodeMemStat>());
    }
    if (allocators.empty()) {
        allocators[0] = std::make_shared<NUMAAllocator>(0);
        node_mem_stats.emplace(0, std::make_unique<NodeMemStat>());
    }
}

// 创建张量
torch::Tensor numa_empty(std::vector<int64_t> shape,
                         int numa_node,
                         c10::optional<torch::ScalarType> dtype,
                         c10::optional<torch::Layout> layout) {
    if (allocators.empty()) {
        init_numa_allocators();
    }
    int requested = numa_node;
    if (allocators.find(numa_node) == allocators.end()) {
        numa_node = allocators.begin()->first;
        std::cerr << "Warning: Requested NUMA node " << requested
                  << " not available. Using node " << numa_node << " instead.\n";
    }
    auto scalar_type = dtype.value_or(torch::kFloat32);
    auto options = torch::TensorOptions()
        .dtype(scalar_type)
        .layout(layout.value_or(torch::kStrided))
        .device(torch::kCPU);

    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    size_t nbytes = numel * c10::elementSize(scalar_type);

    c10::DataPtr data = allocators[numa_node]->allocate(nbytes);

    // 预触碰(按页)保证 mempolicy 查询一致
    const size_t page = 4096;
    uint8_t* base = static_cast<uint8_t*>(data.get());
    for (size_t off = 0; off < nbytes; off += page) base[off] = 0;

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        std::move(data),
        allocators[numa_node].get(),
        /*resizable=*/false);
    c10::Storage storage(storage_impl);

    at::Tensor tensor = at::empty({0}, options);

    // 计算连续 strides
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto* impl = tensor.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(storage);
    impl->set_storage_offset(0);
    impl->set_sizes_and_strides(shape, strides);
    return tensor;
}

// 查询张量所在 NUMA 节点
int get_numa_node(torch::Tensor tensor) {
    if (!tensor.is_cpu()) return -1;
    void* ptr = tensor.data_ptr();
    int node = -1;
    if (get_mempolicy(&node, nullptr, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
        return -1;
    }
    return node;
}

// 复制到指定 NUMA 节点
torch::Tensor to_numa(torch::Tensor tensor, int numa_node) {
    if (!tensor.is_cpu()) {
        throw std::runtime_error("Only CPU tensors can be moved to NUMA nodes");
    }
    auto dst = numa_empty(tensor.sizes().vec(),
                          numa_node,
                          tensor.scalar_type(),
                          tensor.layout());
    dst.copy_(tensor);
    return dst;
}

// 节点内存统计
std::tuple<size_t,size_t> get_node_mem_stats(int node) {
    if (allocators.empty()) init_numa_allocators();
    auto it = node_mem_stats.find(node);
    if (it == node_mem_stats.end()) return {0,0};
    return { it->second->current.load(std::memory_order_relaxed),
             it->second->peak.load(std::memory_order_relaxed) };
}

// 可选：重置峰值
void reset_node_peak(int node) {
    auto it = node_mem_stats.find(node);
    if (it != node_mem_stats.end()) {
        it->second->peak.store(it->second->current.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("numa_empty", &numa_empty, "Create empty tensor on NUMA node",
          py::arg("shape"), py::arg("numa_node"),
          py::arg("dtype") = py::none(),
          py::arg("layout") = py::none());
    m.def("get_numa_node", &get_numa_node, "Get NUMA node of tensor");
    m.def("to_numa", &to_numa, "Move tensor to NUMA node");
    m.def("init_numa_allocators", &init_numa_allocators, "Initialize NUMA allocators");
    m.def("get_node_mem_stats", &get_node_mem_stats, "Get (current, peak) bytes for a NUMA node",
          py::arg("node"));
    m.def("reset_node_peak", &reset_node_peak, "Reset peak to current", py::arg("node"));
}