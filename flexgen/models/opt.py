from pathlib import Path
import numpy as np
from flexgen.models.base import BaseModelLayer, BaseTransformerLayer
from flexgen.models.utils import init_weight_list
# from flex_model import init_weight_list, ValueHolder

class OPTInputEmbed(BaseModelLayer):
    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        s = self.config.max_seq_len
        
        # OPT 有词嵌入和绝对位置嵌入
        weight_specs = [
            ((v, h), "embed_tokens", Path(converted_path) / "embed_tokens"),
            ((s + 2, h), "embed_positions", Path(converted_path) / "embed_positions")
        ]
        
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class OPTSelfAttention(BaseModelLayer):
    def __init__(self, config, env, policy, layer_id: int):
        super().__init__(config, env, policy)
        self.layer_id = layer_id

    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        
        weight_specs = [
            ((h, h), "attn_q_proj", p / f"attn_q_proj.{self.layer_id}.npy"),
            ((h, h), "attn_k_proj", p / f"attn_k_proj.{self.layer_id}.npy"),
            ((h, h), "attn_v_proj", p / f"attn_v_proj.{self.layer_id}.npy"),
            ((h, h), "attn_o_proj", p / f"attn_o_proj.{self.layer_id}.npy"),
            ((h,), "attn_norm", p / f"attn_norm.{self.layer_id}.npy"),
            # 注意: OPT 的权重通常带有 bias，这里为了简化省略了。
            # 实际实现中需要从 config.layer_name_map 获取正确的带 .bias 的文件名
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class OPTMLP(BaseModelLayer):
    def __init__(self, config, env, policy, layer_id: int):
        super().__init__(config, env, policy)
        self.layer_id = layer_id

    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        
        weight_specs = [
            ((4 * h, h), "mlp_fc1", p / f"mlp_fc1.{self.layer_id}.npy"),
            ((h, 4 * h), "mlp_fc2", p / f"mlp_fc2.{self.layer_id}.npy"),
            ((h,), "mlp_norm", p / f"mlp_norm.{self.layer_id}.npy"),
            # 同样，bias 项被省略
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class OPTTransformerLayer(BaseTransformerLayer):
    def __init__(self, config, env, policy, layer_id: int):
        super().__init__(config, env, policy, layer_id)
        self.attention = OPTSelfAttention(config, env, policy, layer_id)
        self.mlp = OPTMLP(config, env, policy, layer_id)