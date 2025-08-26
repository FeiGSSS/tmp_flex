import numpy as np
import os, torch, dataclasses, json
from pathlib import Path
from typing import Union, Optional, Any, Sequence, List
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


from flexgen.utils import str_to_dtype, torch_dtype_to_np_dtype, np_dtype_to_torch_dtype, DUMMY_WEIGHT, GB
from flexgen.compression import CompressionConfig

# DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    w_numa_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    cache_numa_percent: float
    act_gpu_percent: float
    act_cpu_percent: float
    act_numa_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent - self.w_numa_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent - self.cache_numa_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent - self.act_numa_percent


@dataclasses.dataclass(frozen=True)
class Task:
    """A generation task."""
    inputs: Union[np.array, List[List[int]]]
    prompt_len: int
    gen_len: int
    cut_gen_len: Optional[int]

    do_sample: bool
    temperature: float
    stop: Optional[int]

    logits: bool = False  # Whether to return logits for each token


@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    mixed: Any = None
    numa: Any = None

    # @classmethod
    # def create(cls, offload_dir):
    #     # fix recursive import
    #     from flexgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
    #     gpu = TorchDevice("cuda:0")
    #     cpu = TorchDevice("cpu")
    #     disk = TorchDisk(offload_dir)
    #     numa = TorchDevice("numa")
    #     return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]), numa=numa)

    def close_copy_threads(self):
        self.disk.close_copy_threads()




# 定义一个常见的仅解码器模型架构列表，用于判断
# 这个列表可以根据需要进行扩充
DECODER_ONLY_ARCHITECTURES = [
    "LlamaForCausalLM",  # 添加LLaMA支持
    "GPT2LMHeadModel",
    "GPTJForCausalLM",
    "GPTNeoXForCausalLM",
    "OPTForCausalLM",
    "BloomForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "GemmaForCausalLM",
]

def get_tokenizer(
    model_path: str,
    use_fast_tokenizer: bool = True,
):
    gguf_files = None
    tokenizer_path_or_file = None
    kwargs = {}
    path = Path(model_path)

    if path.is_file() and path.suffix == ".gguf":
        gguf_files = [path]
    else:
        gguf_files = list(path.glob("*.gguf"))
        
    if gguf_files:
        kwargs['gguf_file'] = gguf_files[0]
        # return AutoTokenizer.from_pretrained(model_path, **kwargs)
    
    kwargs["use_fast"] = use_fast_tokenizer
    config_path = None
    if gguf_files:
        if len(gguf_files) > 1:
            print(f"发现多个 .gguf 文件 默认加载第一个 .gguf 文件: {gguf_files[0]}")
        tokenizer_path_or_file = gguf_files[0]
    else:
        tokenizer_path_or_file = path

    # --- 加载分词器和配置 ---
    if not gguf_files:
        config_path = tokenizer_path_or_file / "config.json"
        if not os.path.isfile(config_path):
            raise ValueError(f"请检查路径 未找到任何 .gguf 文件或 config.json 文件")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
        architectures = config.get("architectures", [])
        is_decoder_only = any(arch in DECODER_ONLY_ARCHITECTURES for arch in architectures)

    if is_decoder_only or gguf_files:
        if is_decoder_only:
             print(f"检测到仅解码器架构: {architectures[0]}。")
        else: # GGUF
             print("检测到 GGUF 模型，假定为仅解码器架构。")
        print("设置 padding_side='left'。")
        kwargs['padding_side'] = 'left'
    else:
        print(f"检测到非仅解码器或未知架构: {architectures}。使用默认填充设置。")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_file, **kwargs)

    # --- 设置 Pad Token ---
    if getattr(tokenizer, 'padding_side', 'right') == 'left':
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                print("分词器的 pad_token 未设置。使用 eos_token 作为 pad_token。")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                print("警告: 无法自动设置 pad_token，因为 eos_token 也不存在。")
    
    print("\n分词器加载成功！")
    print(f" - 分词器类: {tokenizer.__class__.__name__}")
    print(f" - 词汇表大小: {tokenizer.vocab_size}")
    print(f" - Padding Side: {tokenizer.padding_side}")
    print(f" - Pad Token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f" - BOS Token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f" - EOS Token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    return tokenizer

def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent, policy.w_numa_percent]
    dev_choices = [env.disk, env.cpu, env.gpu, env.numa]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    flag = 0
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, name, filename, dtype = weight_specs[i]
        # if isinstance(dtype, str):
        #     dtype, flag = str_to_dtype[dtype], 1
        # else:
        #     dtype, flag = np_dtype_to_torch_dtype[dtype], 0
        # dtype, flag= str_to_dtype[dtype], 1 if isinstance(dtype, str) else dtype

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            dtype = str_to_dtype[dtype]
            # print(flag)
            
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in str(filename):
                weight.load_from_torch_file(weight_specs[i][2])
            else:
                weight.load_from_torch(torch.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_torch_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_torch(torch.ones(x.shape, x.dtype))

        ret.append(weight)
    return ret

def get_weight_tuple(h, path:str, prefix:str, weight_map:dict):
    """
    将特定结构的字典转换为元组列表。

    参数:
    h (tuple(int or float)): 一个可配置的参数，用于构建元组中第一个元素，为该层size，一次为 hidden、vocab、seqlen。
    path (str): 一个路径字符串，用于构建元组中第三个元素，为保存的 numpy 格式的权重的位置。
    prefix (str): 一个前缀字符串，用于构建元组中第二个元素名字，为名字。
    weight_map (dict): 源字典。键是字符串，值是列表，为保存的 numpy 格式的权重的位置。

    返回:
    list: 包含转换后元组的列表。
    """
    # 初始化一个空列表，用于存储最终结果
    result_list = []
    
    def fn(path:str, prefix:str, key:str, value:list):
        weight_bias = ['weight', 'bias']
        tmp = []
        for idx in range(len(value)):
            name = f"{prefix}_{key}_{weight_bias[idx]}"
            if "norm" in key:
                size = (h,)
            else:
                size = (h, h)
            tmp.append((size, name, path / value[idx], np.float16))
        return tmp
    for key, value in weight_map.items():
        if isinstance(value, dict):
            tmp_result = get_weight_tuple(h, path, prefix, value)
        else:
            tmp_result = fn(path, prefix, key, value)
        result_list.extend(tmp_result)
    return result_list


if __name__ == "__main__":
    dic = {
      "attention": {
        "q_proj": [
          "model.decoder.layers.0.self_attn.q_proj.weight",
          "model.decoder.layers.0.self_attn.q_proj.bias"
        ],
        "k_proj": [
          "model.decoder.layers.0.self_attn.k_proj.weight",
          "model.decoder.layers.0.self_attn.k_proj.bias"
        ],
        "v_proj": [
          "model.decoder.layers.0.self_attn.v_proj.weight",
          "model.decoder.layers.0.self_attn.v_proj.bias"
        ],
        "o_proj": [
          "model.decoder.layers.0.self_attn.out_proj.weight",
          "model.decoder.layers.0.self_attn.out_proj.bias"
        ],
        "norm": [
          "model.decoder.layers.0.self_attn_layer_norm.weight",
          "model.decoder.layers.0.self_attn_layer_norm.bias"
        ]
      },
      "mlp": {
        "fc1": [
          "model.decoder.layers.0.fc1.weight",
          "model.decoder.layers.0.fc1.bias"
        ],
        "fc2": [
          "model.decoder.layers.0.fc2.weight",
          "model.decoder.layers.0.fc2.bias"
        ],
        "norm": [
          "model.decoder.layers.0.final_layer_norm.weight",
          "model.decoder.layers.0.final_layer_norm.bias"
        ]
      }
    }
    c = {"lm_head": [
    "lm_head.weight"
  ]}
    
    path = "/shared/model/deepseek-r1-llama-32b/deepseek-r1-llama-32b-flexgen-np"
    from pathlib import Path
    path = Path(path)
    print(get_weight_tuple(h=1024, prefix='atten', path=path, weight_map=dic))
    # print(get_weight_tuple(h=1024, prefix='lm_head', path=path, weight_map=c))
    pass