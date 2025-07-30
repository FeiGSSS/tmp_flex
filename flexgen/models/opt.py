from pathlib import Path
import numpy as np
from flexgen.models.base import BaseModelLayer, BaseTransformerLayer
from flexgen.models.utils import init_weight_list, get_weight_tuple
# from flex_model import init_weight_list, ValueHolder

class OPTInputEmbed(BaseModelLayer):
    def __init__(self, config, env, policy, weight_map):
        super().__init__(config, env, policy, weight_map=weight_map)   
    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        s = self.config.max_seq_len
        p = Path(converted_path)
        
        # OPT 有词嵌入和绝对位置嵌入
        weight_specs = [
            ((v, h), "embed_tokens", p / self.weight_map["embed_tokens"]),
            ((s + 2, h), "embed_positions", p / self.weight_map["embed_positions"])
        ]
        
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class OPTSelfAttention(BaseModelLayer):
    def __init__(self, config, env, policy, weight_map):
        super().__init__(config, env, policy, weight_map=weight_map)   

    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        
        weight_specs = get_weight_tuple(h=h, path=p, prefix='attn', weight_map=self.weight_map)
        # weight_specs = [
        #     ((h, h), "attn_q_proj_weight", p / self.weight_map["q_proj"][0]),
        #     ((h, h), "attn_q_proj_bias", p / self.weight_map["q_proj"][1]),
        #     ((h, h), "attn_k_proj_weight", p / self.weight_map["k_proj"][0]),
        #     ((h, h), "attn_k_proj_bias", p / self.weight_map["k_proj"][1]),
        #     ((h, h), "attn_v_proj_weight", p / self.weight_map["v_proj"][0]),
        #     ((h, h), "attn_v_proj_bias", p / self.weight_map["v_proj"][1]),
        #     ((h, h), "attn_o_proj_weight", p / self.weight_map["o_proj"][0]),
        #     ((h, h), "attn_o_proj_bias", p / self.weight_map["o_proj"][1]),
        #     ((h,), "attn_norm_weight", p / self.weight_map["norm"][0]),
        #     ((h,), "attn_norm_bias", p / self.weight_map["norm"][1]),
        # ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class OPTMLP(BaseModelLayer):
    def __init__(self, config, env, policy, weight_map):
        super().__init__(config, env, policy, weight_map=weight_map) 

    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        weight_specs = get_weight_tuple(h=h, path=p, prefix='mlp', weight_map=self.weight_map)
        # weight_specs = [
        #     ((4 * h, h), "mlp_fc1_weight", p / self.weight_map["fc1"][0]),
        #     ((4 * h, h), "mlp_fc1_bias", p / self.weight_map["fc1"][1]),
        #     ((h, 4 * h), "mlp_fc2_weight", p / self.weight_map["fc2"][0]),
        #     ((h, 4 * h), "mlp_fc2_bias", p / self.weight_map["fc2"][1]),
        #     ((h,), "mlp_norm_weight", p / self.weight_map["norm"][0]),
        #     ((h,), "mlp_norm_bias", p / self.weight_map["norm"][1]),
        # ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class OPTTransformerLayer(BaseTransformerLayer):
    def __init__(self, config, env, policy, layer_id: int, weight_map: dict):
        layer_map = weight_map['layers'][layer_id]
        super().__init__(config, env, policy, layer_id, layer_map)
        self.attention = OPTSelfAttention(config, env, policy, layer_id, layer_map['attention'])
        self.mlp = OPTMLP(config, env, policy, layer_id, layer_map['mlp'])


class OPTOutputEmbed(BaseModelLayer):
    def __init__(self, config, env, policy, weight_map):
        super().__init__(config, env, policy, weight_map=weight_map)   
    def init_weight(self, weight_home, converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        p = Path(converted_path)
        # weight_specs = get_weight_tuple(h=h, path=p, prefix='output', weight_map=self.weight_map)
        weight_specs = [
            # w_ln
            ((h,), "final_norm_weight", p / "model.decoder.final_layer_norm.weight"),
            # b_ln
            ((h,), "final_norm_bias", p / "model.decoder.final_layer_norm.bias"),
            ((v, h), "lm_head", p / self.weight_map['lm_head']),
        ]

        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)