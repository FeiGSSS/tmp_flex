from pathlib import Path
import numpy as np
from flexgen.utils import ValueHolder

# 假设这些依赖项在您的主项目中可用
# from flex_model_config import FlexModelConfig
# from flex_model import Policy, init_weight_list

class BaseModelLayer:
    """所有模型层的基类，共享通用属性。"""
    def __init__(self, config, env, policy, weight_map: dict):
        self.config = config
        self.env = env
        self.policy = policy
        self.weight_map = weight_map
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, converted_path: str):
        """
        为该层初始化权重。
        每个子类都必须实现这个方法来定义自己的权重规格。
        """
        raise NotImplementedError("Each model layer must implement its own init_weight method.")
    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), np.float16


class BaseTransformerLayer(BaseModelLayer):
    """Transformer层的基类。"""
    def __init__(self, config, env, policy, layer_id: int, weight_map: dict):
        super().__init__(config, env, policy, weight_map=weight_map)
        self.layer_id = layer_id
        # 子类将负责实例化具体的 attention 和 mlp 层
        self.attention = None
        self.mlp = None

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)
    def init_weight(self, weight_home, converted_path: str):
        """组合 attention 和 mlp 层的权重初始化。"""
        # ValueHolder 是 FlexGen 中用于管理张量的辅助类
        home1, home2 = ValueHolder(), ValueHolder() 
        self.attention.init_weight(home1, converted_path)
        self.mlp.init_weight(home2, converted_path)
        weight_home.store((home1, home2))
