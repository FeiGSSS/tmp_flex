from pathlib import Path
from flexgen.models.base import BaseModelLayer, BaseTransformerLayer
from flexgen.models.utils import init_weight_list
# from flex_model import init_weight_list, ValueHolder

class LLaMAInputEmbed(BaseModelLayer):
    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        
        # LLaMA-like 模型只有词嵌入 (位置编码由 RoPE 在计算中动态生成)
        weight_specs = [
            ((v, h), "embed_tokens", Path(converted_path) / "embed_tokens.npy")
        ]
        
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class LLaMASelfAttention(BaseModelLayer):
    def __init__(self, config, env, policy, layer_id: int):
        super().__init__(config, env, policy)
        self.layer_id = layer_id

    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        
        # LLaMA-like 模型通常没有 bias
        weight_specs = [
            ((h, h), "attn_q_proj", p / f"attn_q_proj.{self.layer_id}.npy"),
            ((h, h), "attn_k_proj", p / f"attn_k_proj.{self.layer_id}.npy"),
            ((h, h), "attn_v_proj", p / f"attn_v_proj.{self.layer_id}.npy"),
            ((h, h), "attn_o_proj", p / f"attn_o_proj.{self.layer_id}.npy"),
            ((h,), "attn_norm", p / f"attn_norm.{self.layer_id}.npy"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class LLaMAMLP(BaseModelLayer):
    def __init__(self, config, env, policy, layer_id: int):
        super().__init__(config, env, policy)
        self.layer_id = layer_id

    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        
        # SwiGLU MLP 有三个权重矩阵
        weight_specs = [
            ((4 * h, h), "mlp_gate_proj", p / f"mlp_gate_proj.{self.layer_id}.npy"),
            ((4 * h, h), "mlp_up_proj", p / f"mlp_up_proj.{self.layer_id}.npy"),
            ((h, 4 * h), "mlp_down_proj", p / f"mlp_down_proj.{self.layer_id}.npy"),
            ((h,), "mlp_norm", p / f"mlp_norm.{self.layer_id}.npy"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class LLaMATransformerLayer(BaseTransformerLayer):
    def __init__(self, config, env, policy, layer_id: int):
        super().__init__(config, env, policy, layer_id)
        self.attention = LLaMASelfAttention(config, env, policy, layer_id)
        self.mlp = LLaMAMLP(config, env, policy, layer_id)