# ==============================================================================
# FILE: convert_weights.py
# 描述: 统一的权重转换器，能处理 .bin, .safetensors, .gguf 文件。
# ==============================================================================
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from torch import load as load_bin
from gguf import GGUFReader
import shutil

# 导入上面定义的配置系统
from flexgen.models.config import FlexModelConfigFactory

def load_state_dict_and_config(model_path: str):
    """智能加载权重和元数据/配置
        return的数据格式为(dict, dict)
    """
    path = Path(model_path)
    state_dict = {}

    if path.is_file() and path.suffix == ".gguf":
        gguf_files = [path]
    else:
        gguf_files = list(path.glob("*.gguf"))
    
    if gguf_files:
        if len(gguf_files) > 1:
            print(f"Warning: Multiple GGUF files found, only loading the first one: {gguf_files[0]}")
        print(f"Found {len(gguf_files)} GGUF file, the first gguf file is: {gguf_files[0]}")
        reader = GGUFReader(gguf_files[0], 'r')
        tensors = {tensor.name: tensor.data for tensor in reader.tensors}
        metadata = {field.name: field.data[field.data_idx] for field in reader.fields.values()}
        return tensors, metadata

    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    config = json.load(open(config_path))

    safetensors_files = list(path.glob("*.safetensors"))

    if safetensors_files:
        print(f"Found {len(safetensors_files)} safetensors file, the first safetensors_files is : {safetensors_files[0]}")
        for f_path in tqdm(safetensors_files, desc="Loading safetensors shards"):
            state_dict.update(load_safetensors(f_path))
        return state_dict, config
        # return load_safetensors(safetensors_files[0]), config

    bin_files = list(path.glob("*.bin"))
    if bin_files:
        print(f"Found {len(bin_files)} bin file, the first bin file is: {bin_files[0]}")
        for f_path in tqdm(bin_files, desc="Loading bin shards"):
            state_dict.update(load_bin(f_path, map_location="cpu"))
        return state_dict, config
        # return load_bin(bin_files[0], map_location="cpu"), config
    
    raise FileNotFoundError(f"No valid model files (.bin, .safetensors, .gguf) found in {model_path}")

def save_param(param, output_dir, flexgen_name, data_dtype):
    """将张量保存为NumPy格式"""
    # GGUF 张量已经是 numpy, PyTorch 张量需要转换
    # print(param.dtype)
    param_path = Path(output_dir) / flexgen_name
    data_dtype[flexgen_name] = str(param.dtype)
    # if isinstance(param, torch.Tensor):
    #     if param.dtype == torch.bfloat16:
    #         # print(param.dtype)
    #         param = param.type(torch.float32)
    #     param_np = param.cpu().detach().numpy()
    # else:
    #     param_np = param

    # # FlexGen的权重通常是FP16
    # if param_np.dtype != np.float32:
    #     param_np = param_np.astype(np.float32)
    torch.save(param, param_path)

    # param_path = Path(output_dir) / flexgen_name
    # with open(param_path, "wb") as f:
    #     np.save(f, param.cpu().detach().numpy())


def convert(model_path: str, output_path: str):
    """执行完整的模型转换流程"""
    # 1. 智能加载权重和配置
    state_dict, metadata_or_config = load_state_dict_and_config(model_path)
    
    # 2. 创建FlexModelConfig
    if "model_type" in metadata_or_config: # Hugging Face
        config = FlexModelConfigFactory.from_hf_config(metadata_or_config)
    else: # GGUF
        config = FlexModelConfigFactory.from_gguf_metadata(metadata_or_config)

    print(f"Converting model type: {config.model_type}")
    
    # 3. 遍历并保存权重
    # 创建反向映射，便于查找
    reverse_map = {v: k for k, v in config.layer_name_map.items()}

    # --- 建立权重映射表 方便后续模型读取：初始化 weight_map ---
    weight_map = {
        "layers": [
            {"attention": {}, "mlp": {}} for _ in range(config.num_hidden_layers)
        ]
    }
    data_dtype = {}
    for raw_name, param in tqdm(state_dict.items(), desc="Converting Tensors"):

        save_param(param, output_path, raw_name, data_dtype)
            # shared embedding
        # if "embed_tokens.weight" in raw_name:
        #     shutil.copy(output_path, output_path.replace(
        #         "decoder.embed_tokens.weight", "lm_head.weight"))
        
    # --- 步骤 2: 构建并保存 weight_map.json ---
    print("Building weight map manifest...")
    weight_map = {
        "layers": [
            {"attention": {}, "mlp": {}} for _ in range(config.num_hidden_layers)
        ]
    }
    
    # 遍历 config 中的模板，以构建结构化的清单
    for flexgen_name_template, raw_name_template in tqdm(config.layer_name_map.items(), desc="Building Manifest"):
        # 检查是否是层内权重 (包含占位符 {i})
        if "{i}" in raw_name_template:
            for i in range(config.num_hidden_layers):
                # 新增：检查新是否存在专家占位符 {j}
                if "{j}" in raw_name_template:
                    num_experts = getattr(config, "num_local_experts", 64) 
                    for j in range(num_experts):
                        raw_name = raw_name_template.format(i=i, j=j)
                        if raw_name in state_dict:
                            component = 'mlp'
                            standard_name = f"experts.{j}." + flexgen_name_template.split('_', 2)[2]
                            
                            # 检查是否存在 bias
                            if raw_name.replace('weight', 'bias') in state_dict:
                                weight_map['layers'][i][component][standard_name] = [raw_name, raw_name.replace('weight', 'bias'), data_dtype[raw_name], data_dtype[raw_name.replace('weight', 'bias')]]
                            else:
                                weight_map['layers'][i][component][standard_name] = [raw_name, data_dtype[raw_name]]
                else:
                    # 非 MoE 层处理逻辑
                    raw_name = raw_name_template.format(i=i)
                    if raw_name in state_dict:
                        component = 'attention' if 'attn' in flexgen_name_template else 'mlp'
                        standard_name = flexgen_name_template.split('_', 1)[1]
                        
                        if raw_name.replace('weight', 'bias') in state_dict:
                            weight_map['layers'][i][component][standard_name] = [raw_name, raw_name.replace('weight', 'bias'), data_dtype[raw_name], data_dtype[raw_name.replace('weight', 'bias')]]
                        else:
                            weight_map['layers'][i][component][standard_name] = [raw_name, data_dtype[raw_name]]

                
                
                # # 生成具体的原始权重名
                # raw_name = raw_name_template.format(i=i)
                # # 确认这个权重确实存在
                # if raw_name in state_dict:
                #     component = 'attention' if 'attn' in flexgen_name_template else 'mlp'
                #     standard_name = flexgen_name_template.split('_', 1)[1]
                #     # 在清单中记录此权重的原始文件名
                #     if raw_name.replace('weight', 'bias') in state_dict:
                #         weight_map['layers'][i][component][standard_name] = [raw_name, raw_name.replace('weight', 'bias'), data_dtype[raw_name], data_dtype[raw_name.replace('weight', 'bias')]]
                #     else:
                #         weight_map['layers'][i][component][standard_name] = [raw_name, data_dtype[raw_name]]
        else: # 处理非循环/全局权重
            raw_name = raw_name_template
            if raw_name in state_dict:
                # 在清单中记录此权重的原始文件名
                if raw_name.replace('weight', 'bias') in state_dict:
                    weight_map[flexgen_name_template] = [raw_name, raw_name.replace('weight', 'bias'), data_dtype[raw_name], data_dtype[raw_name.replace('weight', 'bias')]]
                else:
                    weight_map[flexgen_name_template] = [raw_name, data_dtype[raw_name]]
        # print(weight_map[flexgen_name_template]);exit()

    # --- 步骤 3: 保存 config 和 weight_map ---
    with open(Path(output_path) / "weight_map.json", "w") as f:
        json.dump(weight_map, f, indent=2)
    config_save_path = Path(output_path) / f"flexgen_config.pkl"
    with open(config_save_path, "wb") as f:
        pickle.dump(config, f)
        # json.dump(config, f, indent=2)
    print(f"FlexGen config saved to {config_save_path}")
