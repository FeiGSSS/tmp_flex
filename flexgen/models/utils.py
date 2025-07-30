import numpy as np
import os

from flexgen.utils import torch_dtype_to_np_dtype


DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

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
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

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
            tmp.append((size, name, path / value[idx]))
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