from pathlib import Path
import numpy as np
import torch
from typing import Any, Union

from flexgen.models.base import BaseModelLayer, BaseTransformerLayer, BaseModel
from flexgen.models.utils import init_weight_list, get_weight_tuple
from flexgen.utils import (ValueHolder, ExecutionEnv, Task, 
                           array_1d, array_2d, array_3d, array_4d)
from flexgen.models.config import FlexModelConfig
from flexgen.flex_model import Policy
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink, TorchNuma,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)

# from flex_model import init_weight_list, ValueHolder
fix_recursive_import()


class OPTInputEmbed(BaseModelLayer):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict):
        super().__init__(config, env, policy, weight_map=weight_map)   
    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
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
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:Any[int, float]):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64
    
    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i: Any[int, float], 
                k: Any[int, float]):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h


class OPTOutputEmbed(BaseModelLayer):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict):
        super().__init__(config, env, policy, weight_map=weight_map)   
    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
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
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:Any[int, float]):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))
    
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i: Any[int, float], 
                k: Any[int, float]):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h

class OPTSelfAttention(BaseModelLayer):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict):
        super().__init__(config, env, policy, weight_map=weight_map) 
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)  

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
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
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:Any[int, float]):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
    
    def init_cache_one_gpu_batch(self, cache_home:ValueHolder):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        elif self.policy.cache_numa_percent == 100:
            device = self.env.numa
        else:
            raise NotImplementedError()

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)
    
    def load_cache(self, 
                   cache_home:ValueHolder, 
                   cache_read_buf:ValueHolder, 
                   i:Any[int, float]):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")
        
    def store_cache(self, 
                    cache_home:ValueHolder, 
                    cache_write_buf:ValueHolder, 
                    i:Any[int, float]):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), np.float16

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:Any[int, float], 
                k:Any[int, float]):
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h

class OPTMLP(BaseModelLayer):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict):
        super().__init__(config, env, policy, weight_map=weight_map) 

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
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

    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:Any[int, float]):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
    
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), np.float16
    
    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:Any[int, float], 
                k:Any[int, float]):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h


class OPTTransformerLayer(BaseTransformerLayer):
    def __init__(self, config, env, policy, layer_id: int, weight_map: dict):
        layer_map = weight_map[layer_id]
        super().__init__(config, env, policy, layer_id, layer_map)
        self.attention = OPTSelfAttention(config, env, policy, layer_id, layer_map['attention'])
        self.mlp = OPTMLP(config, env, policy, layer_id, layer_map['mlp'])

    def init_weight(self, 
                    weight_home, 
                    path):
        # if self.policy.sep_layer:
        #     # self.attention.init_weight(weight_home, path)
        #     # self.mlp.init_weight(weight_home, path)
        #     # return
        #     raise ValueError(f"Not implemented yet for sep_layer=True")
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:Any[int, float]):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home:ValueHolder):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, 
                   cache_home:ValueHolder, 
                   cache_read_buf:ValueHolder, 
                   i:Any[int, float]):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, 
                    cache_home:ValueHolder, 
                    cache_write_buf:ValueHolder, 
                    i:Any[int, float]):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:Any[int, float], 
                k:Any[int, float]):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)




class OptModel(BaseModel):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict):
        super().__init__(config, env, policy, weight_map=weight_map) 
        self.config = config
        self.env = env
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.weight_map = weight_map
        # InputEmbed, TransformerLayer, OutputEmbed = get_model_architecture(self.config.model_type)
        self.layers.append(OPTInputEmbed(self.config, self.env, self.policy, self.weight_map))
        for layer_id in range(self.config.num_hidden_layers):
            if self.policy.sep_layer:
                self.layers.append(OPTSelfAttention(self.config, self.env, self.policy, layer_id, self.weight_map['layers'][layer_id]['attention']))
                self.layers.append(OPTMLP(self.config, self.env, self.policy, layer_id, self.weight_map['layers'][layer_id]['mlp']))
            else:
                self.layers.append(OPTTransformerLayer(self.config, self.env, self.policy, layer_id, self.weight_map['layers']))
        self.layers.append(OPTOutputEmbed(self.config, self.env, self.policy, self.weight_map))


        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        elif self.policy.act_numa_percent == 100:
            self.act_home = self.env.numa
        else:
            raise NotImplementedError()

        self.init_cache_area() # 初始化用于保存权重 激活 cache 的内存区域

        self.task = None
        
        flexgen_weight_path = self.policy.path
        self.init_all_weights()

    

    def init_all_weights(self, flexgen_weight_path):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for layer_id, layer in enumerate(self.layers):
            layer.init_weight(self.weight_home[layer_id], flexgen_weight_path)

    def get_hidden_area(self, cls, i):
        return cls(*i, ValueHolder)
    
    def load_weight(self, 
                    layer:Union[OPTInputEmbed, OPTSelfAttention, OPTMLP, OPTTransformerLayer, OPTOutputEmbed],  
                    i:Any[int, float], 
                    j:Any[int, float], 
                    k:Any[int, float], 
                    overlap:bool =True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                layer.load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            layer.load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()
    
    def init_cache(self, 
                   layer:Union[OPTInputEmbed, OPTSelfAttention, OPTMLP, OPTTransformerLayer, OPTOutputEmbed], 
                   j:Any[int, float], 
                   k:Any[int, float]):
        layer.init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, 
                   layer:Union[OPTInputEmbed, OPTSelfAttention, OPTMLP, OPTTransformerLayer, OPTOutputEmbed],  
                   i:Any[int, float], 
                   j:Any[int, float], 
                   k:Any[int, float], 
                   overlap:bool =True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                layer.load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            layer.load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, 
                    layer:Union[OPTInputEmbed, OPTSelfAttention, OPTMLP, OPTTransformerLayer, OPTOutputEmbed], 
                    i:Any[int, float], 
                    j:Any[int, float], 
                    k:Any[int, float], 
                    overlap:bool=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                layer.store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            layer.store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self,
                #    layer:Union[OPTInputEmbed, OPTSelfAttention, OPTMLP, OPTTransformerLayer, OPTOutputEmbed],  
                   i:Any[int, float], 
                   j:Any[int, float], 
                   k:Any[int, float], ):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            # ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            if self.task.logits:
                ids, logits = self.hidden[i][j][k].pop()
                logits = logits.data.detach().cpu().numpy()
                ids = ids.data.detach().cpu().numpy()
            else:
                ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
                logits = None
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)
    
    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)
    
    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)
    
    def get_logits(self, ):
        self.hidden = self.get_hidden_area(array_3d, ())


    def generate(self, ):
        self.hidden = self.get_hidden_area(array_3d, ())


    def __del__(self):
        self.delete_all_weights()