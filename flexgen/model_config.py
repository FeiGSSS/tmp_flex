# ==============================================================================
# FILE: model_config.py
# 描述: 定义了统一的模型配置系统，是整个框架的基石。
# ==============================================================================
import dataclasses
import json
from typing import Dict, Any

from huggingface_hub import hf_hub_download

@dataclasses.dataclass(init=False, kw_only=True)
class FlexModelConfig:
    """ 
    统一模型架构描述。
    这个类的实例将包含运行一个特定模型所需的所有架构信息和权重命名规则。
    """
    # --- 尺寸参数 ---
    input_dim: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    num_key_value_heads: int  # 用于 GQA/MQA
    rms_norm_eps: float

    # --- 架构定义 ---
    model_type: str # 'llama', 'opt', 'deepseek', etc.
    mlp_type: str  # 'Gated-SwiGLU', 'GELU-MLP' 等
    normalization_type: str # 'RMSNorm', 'LayerNorm'
    positional_embedding_type: str # 'Rotary' (RoPE), 'Absolute'

    # --- 层命名映射字典 ---
    # Key = FlexGen内部标准名, Value = 来源框架中的原始名
    # 使用 {i} 作为层号的占位符
    layer_name_map: Dict[str, str]
    def __init__(self, **kwargs):
            # 1. 先检查并设置所有显式定义的核心字段
            required_fields = [field.name for field in dataclasses.fields(self)]
            for field in required_fields:
                if field not in kwargs:
                    raise TypeError(f"缺少必填参数: {field}")
                setattr(self, field, kwargs[field])  # 设置核心字段

            # 2. 将所有参数（包括核心字段和额外字段）存入实例的__dict__
            # 这样可以通过config.xxx直接访问所有参数
            self.__dict__.update(kwargs) 
    def model_bytes(self):
        h = self.input_dim
        return 	2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2
    

    def get_layer_name(self, flexgen_name: str, layer_idx: int = None) -> str:
        """根据FlexGen标准名和层号获取原始名"""
        raw_name_template = self.layer_name_map.get(flexgen_name)
        if raw_name_template is None:
            raise KeyError(f"FlexGen standard name '{flexgen_name}' not found in layer_name_map.")
        if layer_idx is not None and "{i}" in raw_name_template:
            return raw_name_template.format(i=layer_idx)
        return raw_name_template


class FlexModelConfigFactory:
    """
    配置工厂: 根据不同的输入（Hugging Face config 或 GGUF metadata）创建 FlexModelConfig。
    """
    @staticmethod
    def from_hf_config(config: dict) -> FlexModelConfig:
        model_type = config.get("model_type")
        explicit_params = {
                            "model_type", "hidden_size", "num_attention_heads", 
                            "num_hidden_layers", "vocab_size", "num_key_value_heads",
                            "rms_norm_eps", "mlp_type", "normalization_type",
                            "positional_embedding_type", "layer_name_map", 
                        }
        filtered_config = {k: v for k, v in config.items() if k not in explicit_params}

        # LLaMA, DeepSeek, Qwen2 等现代模型架构高度相似
        if model_type in ["llama", "deepseek", "qwen2", "mistral"]:
            return FlexModelConfig(
                input_dim=config["hidden_size"],
                model_type=model_type,
                hidden_size=config["hidden_size"],
                num_attention_heads=config["num_attention_heads"],
                num_hidden_layers=config["num_hidden_layers"],
                
                vocab_size=config["vocab_size"],
                num_key_value_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
                rms_norm_eps=config["rms_norm_eps"],
                mlp_type='Gated-SwiGLU',
                normalization_type='RMSNorm',
                positional_embedding_type='Rotary',
                layer_name_map={
                    "embed_tokens": "model.embed_tokens.weight",
                    "lm_head": "lm_head.weight",
                    "final_norm": "model.norm.weight",
                    "attn_q_proj": "model.layers.{i}.self_attn.q_proj.weight",
                    "attn_k_proj": "model.layers.{i}.self_attn.k_proj.weight",
                    "attn_v_proj": "model.layers.{i}.self_attn.v_proj.weight",
                    "attn_o_proj": "model.layers.{i}.self_attn.o_proj.weight",
                    "mlp_gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
                    "mlp_up_proj": "model.layers.{i}.mlp.up_proj.weight",
                    "mlp_down_proj": "model.layers.{i}.mlp.down_proj.weight",
                    "attn_norm": "model.layers.{i}.input_layernorm.weight",
                    "mlp_norm": "model.layers.{i}.post_attention_layernorm.weight",
                }, 
                **filtered_config, 
            )
        elif model_type == "opt":
            return FlexModelConfig(
                input_dim=config["hidden_size"],
                model_type=model_type,
                hidden_size=config["hidden_size"],
                num_attention_heads=config["num_attention_heads"],
                num_hidden_layers=config["num_hidden_layers"],
                vocab_size=config["vocab_size"],
                num_key_value_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
                rms_norm_eps=config.get("layer_norm_eps", 1e-5), # OPT uses layer_norm_eps
                mlp_type='GELU-MLP',
                normalization_type='LayerNorm',
                positional_embedding_type='Absolute',
                layer_name_map={
                    "embed_tokens": "model.decoder.embed_tokens.weight",
                    "embed_positions": "model.decoder.embed_positions.weight", # OPT specific
                    "lm_head": "lm_head.weight",
                    # No single final_norm for OPT, norm is inside layers
                    "attn_q_proj": "model.decoder.layers.{i}.self_attn.q_proj.weight",
                    "attn_k_proj": "model.decoder.layers.{i}.self_attn.k_proj.weight",
                    "attn_v_proj": "model.decoder.layers.{i}.self_attn.v_proj.weight",
                    "attn_o_proj": "model.decoder.layers.{i}.self_attn.out_proj.weight",
                    "mlp_fc1": "model.decoder.layers.{i}.fc1.weight", # OPT specific
                    "mlp_fc2": "model.decoder.layers.{i}.fc2.weight", # OPT specific
                    "attn_norm": "model.decoder.layers.{i}.self_attn_layer_norm.weight",
                    "mlp_norm": "model.decoder.layers.{i}.final_layer_norm.weight", # Renamed for consistency
                }, 
                **filtered_config,
            )
        elif model_type == "chatglm":
            # ChatGLM (v2/v3) 的实现细节（如QueryKeyValuen一体的权重）需要特别处理
            # 此处为简化示例，实际需要更复杂的映射和后端计算支持
            raise NotImplementedError("ChatGLM support requires custom handling for its unique architecture.")
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported by this factory.")

    @staticmethod
    def from_gguf_metadata(metadata: Dict[str, Any]) -> FlexModelConfig:
        # 从GGUF元数据中提取信息来构建Config
        try:
            prefix = ".".join(list(metadata.keys())[0].split('.')[:-1])
            return FlexModelConfig(
                model_type=prefix,
                hidden_size=int(metadata[f'{prefix}.embedding_length']),
                num_attention_heads=int(metadata[f'{prefix}.attention.head_count']),
                num_hidden_layers=int(metadata[f'{prefix}.block_count']),
                vocab_size=len(metadata[f'{prefix}.tokenizer.ggml.tokens']),
                num_key_value_heads=int(metadata[f'{prefix}.attention.head_count_kv']),
                rms_norm_eps=float(metadata[f'{prefix}.attention.layer_norm_rms_epsilon']),
                mlp_type='Gated-SwiGLU',
                normalization_type='RMSNorm',
                positional_embedding_type='Rotary',
                layer_name_map={
                    "embed_tokens": "token_embd.weight",
                    "lm_head": "output.weight",
                    "final_norm": "output_norm.weight",
                    "attn_q_proj": "blk.{i}.attn_q.weight",
                    "attn_k_proj": "blk.{i}.attn_k.weight",
                    "attn_v_proj": "blk.{i}.attn_v.weight",
                    "attn_o_proj": "blk.{i}.attn_output.weight",
                    "mlp_gate_proj": "blk.{i}.ffn_gate.weight",
                    "mlp_up_proj": "blk.{i}.ffn_up.weight",
                    "mlp_down_proj": "blk.{i}.ffn_down.weight",
                    "attn_norm": "blk.{i}.attn_norm.weight",
                    "mlp_norm": "blk.{i}.ffn_norm.weight",
                }
            )
        except KeyError as e:
            raise ValueError(f"Could not find required key in GGUF metadata: {e}")

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> FlexModelConfig:
        """从Hugging Face Hub或本地路径加载配置"""
        try:
            config_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
            with open(config_path) as f:
                config_dict = json.load(f)
            return cls.from_hf_config(config_dict)
        except Exception as e:
            raise IOError(f"Could not load config for '{model_name_or_path}': {e}")

