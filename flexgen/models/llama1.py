from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Union, List, Tuple
from transformers.activations import ACT2FN

from flexgen.models.base import BaseModelLayer, BaseTransformerLayer, BaseModel
from flexgen.utils import (ValueHolder, 
                           array_1d, array_2d, array_3d, array_4d)
from flexgen.models.utils import (ExecutionEnv, Task, Policy, 
                                  init_weight_list, get_weight_tuple
                           )
from flexgen.models.config import FlexModelConfig
# from flexgen.flex_model import Policy
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink, TorchNuma, TorchTensor, 
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)

# from flex_model import init_weight_list, ValueHolder
fix_recursive_import()

class LlamaModelComputation:
    """
    此类封装了所有为 Llama 模型定义的计算逻辑。
    代码逻辑主要从 pytorch_backend_llama.py 迁移而来。
    """
    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float = 1e-6):
        input_dtype = x.dtype
        x = x.to(torch.float16)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        return (weight * x).to(input_dtype)

    def precompute_freqs_cis(self, dim: int, end: int, inv_freq: torch.Tensor, theta: float = 10000.0):
        freqs = inv_freq
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def input_embed(self, compute_device: TorchDevice, inputs: TorchTensor, w_token: TorchTensor, pad_token_id: int, donate: list):
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        if donate[0]: inputs.delete()
        
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        return TorchTensor.create_from_torch(token_embed, compute_device)

    def mha(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
            w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor, w_o: torch.Tensor,
            config: FlexModelConfig, freqs_cis: torch.Tensor,
            k_cache_full: torch.Tensor, v_cache_full: torch.Tensor, start_pos: int):
        """MHA for PREFILL stage (seqlen > 1)."""
        bsz, seqlen, h = hidden_states.shape
        n_q_head, n_kv_head = config.n_head, getattr(config, 'num_key_value_heads', config.n_head)
        n_kv_groups = n_q_head // n_kv_head
        head_dim = h // n_q_head
        
        xq = F.linear(hidden_states, w_q).view(bsz, seqlen, n_q_head, head_dim)
        xk = F.linear(hidden_states, w_k).view(bsz, seqlen, n_kv_head, head_dim)
        xv = F.linear(hidden_states, w_v).view(bsz, seqlen, n_kv_head, head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        k_cache_full[:bsz, start_pos : start_pos + seqlen] = xk
        v_cache_full[:bsz, start_pos : start_pos + seqlen] = xv

        keys, values = k_cache_full[:bsz, : start_pos + seqlen], v_cache_full[:bsz, : start_pos + seqlen]
        keys, values = self.repeat_kv(keys, n_kv_groups), self.repeat_kv(values, n_kv_groups)

        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / (head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return F.linear(output, w_o)

    def mha_gen(self, hidden_states: torch.Tensor,
                w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor, w_o: torch.Tensor,
                config: FlexModelConfig, freqs_cis: torch.Tensor,
                past_k: torch.Tensor, past_v: torch.Tensor):
        """MHA for DECODE stage (seqlen = 1)."""
        bsz, seqlen, h = hidden_states.shape
        n_q_head, n_kv_head = config.n_head, getattr(config, 'num_key_value_heads', config.n_head)
        n_kv_groups = n_q_head // n_kv_head
        head_dim = h // n_q_head
                       
        q = F.linear(hidden_states, w_q).view(bsz, seqlen, n_q_head, head_dim)
        k = F.linear(hidden_states, w_k).view(bsz, seqlen, n_kv_head, head_dim)
        v = F.linear(hidden_states, w_v).view(bsz, seqlen, n_kv_head, head_dim)
        
        q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
        keys, values = torch.cat([past_k, k], dim=1), torch.cat([past_v, v], dim=1)
        keys, values = self.repeat_kv(keys, n_kv_groups), self.repeat_kv(values, n_kv_groups)
        
        q, keys, values = q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        
        scores = torch.matmul(q, keys.transpose(2, 3)) / (head_dim ** 0.5)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return F.linear(output, w_o), k, v

    def mlp(self, hidden_states: torch.Tensor,
            w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
            config: FlexModelConfig):
        act_fn = ACT2FN[config.hidden_act]
        gate = F.linear(hidden_states, w_gate)
        up = F.linear(hidden_states, w_up)
        return F.linear(act_fn(gate) * up, w_down)
    
    
    # # SelfAttention Prefill
    # def mha(self, compute_device: TorchDevice, hidden_states: TorchTensor, attention_mask: TorchTensor,
    #         w_q: TorchTensor, w_k: TorchTensor, w_v: TorchTensor, w_o: TorchTensor, w_norm: TorchTensor,
    #         n_head: int, config: FlexModelConfig, donate: list, compress_cache: bool, comp_config: any, 
    #         start_pos:int, k_cache:torch.Tensor, v_cache:torch.Tensor):
        
    #     if w_q.device.device_type == DeviceType.COMPRESSED:
    #         w_q, w_k, w_v, w_o = [x.device.decompress(x) for x in [w_q, w_k, w_v, w_o]]

      
    #     bsz, q_len, h = hidden_states.shape
    #     n_q_head = n_head
    #     n_kv_head = getattr(config, 'num_key_value_heads', n_q_head)
    #     num_key_value_groups = n_q_head // n_kv_head
    #     head_dim = h // n_q_head
        
    #     hidden_data = hidden_states.data
    #     # hidden_norm = self.rms_norm(hidden_data, w_norm.data, config.rms_norm_eps)
        
    #     # shape: (b, s, h)
    #     q = F.linear(hidden_data, w_q.data)
    #     k = F.linear(hidden_data, w_k.data)
    #     v = F.linear(hidden_data, w_v.data)

    #     # 重塑为多头形式
    #     # shape: (b, s, n_head, head_dim)
    #     q = q.view(bsz, q_len, n_q_head, head_dim)
    #     k = k.view(bsz, q_len, n_kv_head, head_dim)
    #     v = v.view(bsz, q_len, n_kv_head, head_dim)

    #     # 应用RoPE
    #     q, k = self.apply_rotary_emb(q, k, freqs_cis=compute_device.rotary_emb_cis[:q_len])
        
    #     # 更新缓存
    #     k_cache = k_cache.to(q)
    #     v_cache = v_cache.to(q)
    #     k_cache[:bsz, start_pos:start_pos+q_len] = k
    #     v_cache[:bsz, start_pos:start_pos+q_len] = v

    #     # 获取所有key和value（包含缓存
    #     keys = k_cache[:bsz, : start_pos+q_len]
    #     values = v_cache[:bsz, : start_pos+q_len]

    #     # 重复kv头以匹配q头数
    #     # shape:(b, s, n_head, head_dim)
    #     # if num_key_value_groups > 1:
    #     #   n_head = n_head*num_key_value_groups
    #     # else:
    #     #     n_head = n_head
    #     keys = self.repeat_kv(keys, num_key_value_groups)
    #     values = self.repeat_kv(values, num_key_value_groups)

    #     # 转置后注意力计算
    #     # shape: (b, n_head, s*, head_dim)
    #     # q: s* is q_len
    #     # keys, values: s* is cache_len + qlen
    #     q = q.transpose(1, 2)
    #     keys = keys.transpose(1, 2)
    #     values = values.transpose(1, 2)
    #     # shape: (b, n_head, s_q, s_keys)
    #     scores = torch.matmul(q, keys.transpose(2, 3)) / (head_dim ** 0.5)

    #     # 计算注意力分数
    #     # mask shape is (b, s_q)
    #     mask = attention_mask.data
    #     if mask is not None:
    #         scores = scores + mask
    #     scores = F.softmax(scores, dim=-1, dtype=scores.float()).type_as(q)
    #     # shape: (b, n_head, s, s)
    #     output = torch.matmul(scores, values)
    #     # shape: (b, s, n_head, s)
    #     output = output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    #     return F.linear(output, w_o.data)
    #     o = w_o.data(output) + hidden_data
        

    #     if donate[0]: hidden_states.delete()
    #     if donate[1]: attention_mask.delete()

    #     if compress_cache:
    #         k_to_cache = compute_device.compressed_device.compress(keys, comp_config)
    #         v_to_cache = compute_device.compressed_device.compress(values, comp_config)
    #     else:
    #         k_to_cache = TorchTensor.create_from_torch(keys, compute_device)
    #         v_to_cache = TorchTensor.create_from_torch(values, compute_device)

    #     return TorchTensor.create_from_torch(o, compute_device), k_to_cache, v_to_cache

    # # SelfAttention Decode
    # def mha_gen(self, compute_device: TorchDevice, hidden_states: TorchTensor, attention_mask: TorchTensor,
    #             w_q: TorchTensor, w_k: TorchTensor, w_v: TorchTensor, w_o: TorchTensor, w_norm: TorchTensor,
    #             n_head: int, config: FlexModelConfig, donate: list, compress_cache: bool, comp_config: any,
    #             k_cache: TorchTensor, v_cache: TorchTensor):

    #     if w_q.device.device_type == DeviceType.COMPRESSED:
    #         w_q, w_k, w_v, w_o = [x.device.decompress(x) for x in [w_q, w_k, w_v, w_o]]

    #     bsz, tgt_s, h = hidden_states.shape
    #     src_s = attention_mask.shape[1]
    #     n_q_head = n_head
    #     n_kv_head = getattr(config, 'num_key_value_heads', n_q_head)
    #     num_key_value_groups = n_q_head // n_kv_head
    #     head_dim = h // n_q_head
                       
    #     residual = hidden_states.data
    #     hidden_norm = self.rms_norm(hidden_states.data, w_norm.data, config.rms_norm_eps)

    #     q = F.linear(hidden_norm, w_q.data)
    #     k = F.linear(hidden_norm, w_k.data)
    #     v = F.linear(hidden_norm, w_v.data)

    #     q = q.view(bsz, tgt_s, n_q_head, head_dim)
    #     k = k.view(bsz, tgt_s, n_kv_head, head_dim)
    #     v = v.view(bsz, tgt_s, n_kv_head, head_dim)

    #     freqs_cis = compute_device.rotary_emb_cis[src_s - 1 : src_s]
    #     q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
    #     q = q.permute(0, 2, 1, 3).reshape(bsz * n_q_head, tgt_s, head_dim)
    #     k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, bsz * n_kv_head, head_dim)
    #     v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, bsz * n_kv_head, head_dim)

    #     if compress_cache:
    #         k_cache_data = k_cache.device.decompress(k_cache)[:src_s-1]
    #         v_cache_data = v_cache.device.decompress(v_cache)[:src_s-1]
    #     else:
    #         k_cache_data = k_cache.data[:src_s-1]
    #         v_cache_data = v_cache.data[:src_s-1]
        
    #     k_all_seq = torch.cat([k_cache_data, k_new], dim=0)
    #     v_all_seq = torch.cat([v_cache_data, v_new], dim=0)
        
    #     k_all = k_all_seq.permute(1, 2, 0)
    #     v_all = v_all_seq.permute(1, 0, 2)
        
    #     if num_key_value_groups > 1:
    #         k_all = k_all.view(bsz, n_kv_head, head_dim, src_s).repeat_interleave(num_key_value_groups, dim=1).view(bsz * n_q_head, head_dim, src_s)
    #         v_all = v_all.view(bsz, n_kv_head, src_s, head_dim).repeat_interleave(num_key_value_groups, dim=1).view(bsz * n_q_head, src_s, head_dim)
    
    #     attn_weights = torch.bmm(q, k_all) / (head_dim ** 0.5)
        
    #     # mask = attention_mask.data.view(bsz, 1, 1, src_s).expand(-1, n_q_head, -1, -1)
    #     # attn_weights = attn_weights.view(bsz, n_q_head, tgt_s, src_s)
    #     # attn_weights = attn_weights + mask
    #     # attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v_all.dtype)
    #     mask = attention_mask.data.view(bsz, 1, 1, src_s).expand(-1, n_q_head, -1, -1)
    #     attn_weights = attn_weights.view(bsz, n_q_head, tgt_s, src_s)
    #     attn_weights = torch.where(mask, attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=compute_device.dev))
        
    #     attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v_all.dtype)
        


    #     value = torch.bmm(attn_weights.view(bsz * n_q_head, tgt_s, src_s), v_all)
    #     value = value.view(bsz, n_q_head, tgt_s, head_dim).transpose(1, 2).reshape(bsz, tgt_s, h)
    #     value = F.linear(value, w_o.data)
        
    #     value.add_(residual)
        
    #     if donate[0]: hidden_states.delete()
    #     if donate[1]: attention_mask.delete()

    #     if compress_cache:
    #         k_new = compute_device.compressed_device.compress(k_new, comp_config)
    #         v_new = compute_device.compressed_device.compress(v_new, comp_config)
    #     else:
    #         k_new = TorchTensor.create_from_torch(k_new, compute_device)
    #         v_new = TorchTensor.create_from_torch(v_new, compute_device)

    #     return TorchTensor.create_from_torch(value, compute_device), k_new, v_new
    # def mlp(self, compute_device: TorchDevice, hidden_states: TorchTensor,
    #         w_gate: TorchTensor, w_up: TorchTensor, w_down: TorchTensor, w_norm: TorchTensor,
    #         config: FlexModelConfig, donate: list):
        
    #     if w_gate.device.device_type == DeviceType.COMPRESSED:
    #         w_gate, w_up, w_down = [x.device.decompress(x) for x in [w_gate, w_up, w_down]]

    #     residual = hidden_states.data
    #     hidden_norm = self.rms_norm(hidden_states.data, w_norm.data, config.rms_norm_eps)
        
    #     act_fn = ACT2FN[config.hidden_act]
    #     gate = F.linear(hidden_norm, w_gate.data)
    #     up = F.linear(hidden_norm, w_up.data)
    #     down = F.linear(act_fn(gate) * up, w_down.data)
        
    #     down.add_(residual)
    #     if donate[0]: hidden_states.delete()
        
    #     return TorchTensor.create_from_torch(down, compute_device)

    def output_embed(self, compute_device: TorchDevice, hidden_states: TorchTensor, w_lm_head: TorchTensor, donate: list, do_sample: bool, temperature: float):
        if w_lm_head.device.device_type == DeviceType.COMPRESSED:
            w_lm_head = w_lm_head.device.decompress(w_lm_head)

        logits = F.linear(hidden_states.data, w_lm_head.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
            
        if donate[0]: hidden_states.delete()
        
        return TorchTensor.create_from_torch(ids, compute_device), TorchTensor.create_from_torch(logits, compute_device)

# Llama 2 模型与 OPT 的主要区别:
# 1. 位置编码: 使用旋转位置编码 (Rotary Positional Embedding, RoPE)，在 Attention 层中应用，而不是独立的输入层。
# 2. 归一化层: 使用 RMSNorm (Root Mean Square Normalization)，而不是 LayerNorm。
# 3. MLP/FFN 层: 使用带 SiLU 激活函数的 SwiGLU，而不是带 ReLU 的标准 MLP。
# 4. 注意力机制: 可能使用分组查询注意力 (Grouped-Query Attention, GQA) 来优化推理。

class LlamaRMSNorm(BaseModelLayer):
    """Llama 的 RMSNorm 层 (仅用于模型末尾的 final_norm)。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LlamaModelComputation):
        super().__init__(config, env, policy, weight_map=weight_map)
        self.computation = computation # 待实现

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        weight_specs = [
            ((h,), "final_norm_weight", p / self.weight_map["final_norm"][0], self.config.dtype),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        (w_norm,) = weight_home.val
        if k == 0:
            dst = self.compute_device
            weight_read_buf.store(w_norm.smart_copy(dst))

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        
        donate = [False] * 2
        h, donate[0] = hidden.val, True
        
        if k == self.policy.num_gpu_batches - 1:
            w_norm, donate[1] = weight_read_buf.pop()
        else:
            w_norm, _ = weight_read_buf.val
        
        h.data = self.computation.rms_norm(h.data, w_norm.data, self.config.rms_norm_eps)
        hidden.val = h


class LlamaInputEmbed(BaseModelLayer):
    """Llama 的输入嵌入层，只包含词嵌入。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LlamaModelComputation):
        super().__init__(config, env, policy, weight_map=weight_map)
        self.computation = computation # 待实现

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        p = Path(converted_path)
        
        weight_specs = [
            ((v, h), "embed_tokens", p / self.weight_map["embed_tokens"][0], self.config.dtype),
        ]
        
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        w_token, = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store(w_token.smart_copy(dst))
    
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64
    

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        
        # TODO: 实现词嵌入的前向计算
        # Compute input embedding
        donate = [False] * 2
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            w_token, donate[1] = weight_read_buf.pop()
        else:
            w_token, _ = weight_read_buf.val

        h = self.computation.input_embed(self.compute_device, h,
            w_token, self.config.pad_token_id, donate)
        hidden.val = h

class LlamaOutputEmbed(BaseModelLayer):
    """Llama 的输出层，将最终的 hidden states 映射到词汇表。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LlamaModelComputation):
        super().__init__(config, env, policy, weight_map=weight_map)
        self.computation = computation # 待实现

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        p = Path(converted_path)
        
        weight_specs = [
            # 根据用户提供的字典，使用 "lm_head" 键
            ((v, h), "lm_head", p / self.weight_map["lm_head"][0], self.config.dtype),
        ]
        
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        (w_lm_head,) = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store(w_lm_head.smart_copy(dst))

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        donate = [False] * 2
        h, donate[0] = hidden.val, True
        
        if k == self.policy.num_gpu_batches - 1:
            w_lm_head, donate[1] = weight_read_buf.pop()
        else:
            w_lm_head, _ = weight_read_buf.val
            
        h, logits = self.computation.output_embed(self.compute_device, h, w_lm_head, donate,
            self.task.do_sample, self.task.temperature)
        if self.task.logits:
            hidden.val = [h, logits]
        else:
            hidden.val = h


class LlamaSelfAttention(BaseModelLayer):
    """Llama 的自注意力层，包含 RoPE。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LlamaModelComputation):
        super().__init__(config, env, policy, weight_map=weight_map)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        self.computation = computation
        
        self.h = self.config.hidden_size
        self.num_heads = self.config.n_head
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.num_heads) # 如果没有就退化为 MHA
        self.head_dim = self.h // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim # 计算 K 和 V 的实际维度
        
        s = getattr(self.config, "max_seq_len", None) or getattr(self.config, "max_position_embeddings", None) or 2048
        shape = (policy.gpu_batch_size, s, self.num_kv_heads, self.head_dim)
        # print(shape, config.dtype)
        # exit()
        self.k_cache = torch.zeros(
                                    shape, 
                                    dtype=torch.float16, 
                                    # device=self.compute_device.dev
                                      ).cuda()
        self.v_cache = torch.zeros(
                                    shape, 
                                    dtype=torch.float16, 
                                    # device=self.compute_device.dev
                                      ).cuda()
        

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
        
        p = Path(converted_path)
        
        weight_specs = [
            ((self.h, self.h), "atten_q_proj", p / self.weight_map["q_proj"][0], self.config.dtype),
            ((self.kv_dim, self.h), "atten_k_proj", p / self.weight_map["k_proj"][0], self.config.dtype),
            ((self.kv_dim, self.h), "atten_v_proj", p / self.weight_map["v_proj"][0], self.config.dtype),
            ((self.h, self.h), "atten_o_proj", p / self.weight_map["o_proj"][0], self.config.dtype),
            # ((self.h,), "atten_norm", p / self.weight_map["norm"][0], self.config.dtype),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        w_q, w_k, w_v, w_o = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            # dst2 = self.compute_device
            weight_read_buf.store((
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_o.smart_copy(dst1),
                # w_norm.smart_copy(dst2),
            ))
    
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

        # Reset the stateful caches for a new generation sequence
        self.k_cache.zero_()
        self.v_cache.zero_()
        # FlexGen's cache mechanism is used to move partial caches for decode
        cache_home.store((
            device.allocate((self.policy.gpu_batch_size, 0, self.num_kv_heads, self.head_dim), self.config.dtype),
            device.allocate((self.policy.gpu_batch_size, 0, self.num_kv_heads, self.head_dim), self.config.dtype)
        ))

        # # cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        # # cache_home.store(cache)
        # n_head = self.config.n_head
        # n_kv_head = getattr(self.config, 'num_key_value_heads', n_head)
        # head_dim = self.config.hidden_size // n_head
        
        # prompt_len, gen_len = self.task.prompt_len, self.task.gen_len
        # gpu_batch_size = self.policy.gpu_batch_size
        
        # shape = (prompt_len + gen_len - 1, gpu_batch_size * n_kv_head, head_dim)
        
        # pin_memory = False
        # k_cache = device.allocate(shape, np.float16, pin_memory=pin_memory)
        # v_cache = device.allocate(shape, np.float16, pin_memory=pin_memory)
        # cache_home.store((k_cache, v_cache))

    def load_cache(self, 
                   cache_home:ValueHolder, 
                   cache_read_buf:ValueHolder, 
                   i:List[Union[int, float]]):
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
                    i:List[Union[int, float]]):
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

    def forward(self, 
                hidden_states: TorchTensor, 
                attention_mask: torch.Tensor, 
                weights: List[TorchTensor], 
                freqs_cis: torch.Tensor, 
                start_pos: int, 
                is_prefill: bool, 
                past_kv: Tuple[TorchTensor, TorchTensor]):
        w_q, w_k, w_v, w_o = [w.data for (w, _) in weights]
        
        if is_prefill:
            attn_output = self.computation.mha(hidden_states.data, attention_mask, w_q, w_k, w_v, w_o, self.config, freqs_cis, self.k_cache, self.v_cache, start_pos)
            return TorchTensor.create_from_torch(attn_output, self.compute_device), None, None
        else:
            past_k, past_v = past_kv
            attn_output, new_k, new_v = self.computation.mha_gen(hidden_states.data, w_q, w_k, w_v, w_o, self.config, freqs_cis, past_k.data, past_v.data)
            return TorchTensor.create_from_torch(attn_output, self.compute_device), TorchTensor.create_from_torch(new_k, self.compute_device), TorchTensor.create_from_torch(new_v, self.compute_device)

    # def forward(self, 
    #             hidden, 
    #             cache_read_buf:ValueHolder, 
    #             weight_read_buf:ValueHolder, 
    #             attention_mask:ValueHolder,
    #             cache_write_buf:ValueHolder, 
    #             i:List[Union[int, float]], 
    #             k:List[Union[int, float]]):
    #     donate = [False] * 7
    #     h, donate[0] = hidden.val, True
        
    #     if k == self.policy.num_gpu_batches - 1:
    #         (w_q, donate[2]), (w_k, donate[3]), (w_v, donate[4]), (w_o, donate[5]), (w_norm, donate[6]), = weight_read_buf.pop()
    #     else:
    #         (w_q, _), (w_k, _), (w_v, _), (w_o, _), (w_norm, _), = weight_read_buf.val

    #     if i == 0:  # prefill
    #         mask, donate[1] = attention_mask.val.smart_copy(self.compute_device)
    #         h, new_k_cache, new_v_cache = self.computation.mha(self.compute_device, h, mask, w_q, w_k, w_v, w_o, w_norm,
    #             self.config.n_head, self.config, donate, self.policy.compress_cache, self.policy.comp_cache_config, 
    #             self.task.prompt_len+i-1 ,self.k_to_cache, self.v_to_cache)
    #         cache_write_buf.store((new_k_cache, new_v_cache))
    #     else: # decode
    #         mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
    #         (k_cache, donate[6]), (v_cache, _) = cache_read_buf.pop()
    #         h, new_k_cache, new_v_cache = self.computation.mha_gen(self.compute_device, h, mask, w_q, w_k, w_v, w_o, w_norm,
    #             self.config.n_head, self.config, donate, self.policy.compress_cache, self.policy.comp_cache_config, k_cache, v_cache)
    #         cache_write_buf.store((new_k_cache, new_v_cache))
        
    #     hidden.val = h

class LlamaMLP(BaseModelLayer):
    """Llama 的 MLP 层，使用 SwiGLU。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LlamaModelComputation):
        super().__init__(config, env, policy, weight_map=weight_map)
        self.computation = computation # 待实现

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        i = self.config.ffn_embed_dim
        p = Path(converted_path)

        weight_specs = [
            ((i, h), "mlp_gate_proj", p / self.weight_map["gate_proj"][0], self.config.dtype),
            ((i, h), "mlp_up_proj", p / self.weight_map["up_proj"][0], self.config.dtype),
            ((h, i), "mlp_down_proj", p / self.weight_map["down_proj"][0], self.config.dtype),
            # ((h,), "mlp_norm", p / self.weight_map["norm"][0], self.config.dtype),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        w_gate, w_up, w_down = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            # dst2 = self.compute_device
            weight_read_buf.store((
                w_gate.smart_copy(dst1),
                w_up.smart_copy(dst1),
                w_down.smart_copy(dst1),
                # w_norm.smart_copy(dst2)
            ))

    def forward(self, 
                hidden_states:TorchTensor, 
                weights:List[TorchTensor]):
        w_gate, w_up, w_down = [w.data for (w, _) in weights]
        output = self.computation.mlp(hidden_states.data, w_gate, w_up, w_down, self.config)
        return TorchTensor.create_from_torch(output, self.compute_device)
    
    # def forward(self, 
    #             hidden, 
    #             cache_read_buf:ValueHolder, 
    #             weight_read_buf:ValueHolder, 
    #             attention_mask:ValueHolder,
    #             cache_write_buf:ValueHolder, 
    #             i:List[Union[int, float]], 
    #             k:List[Union[int, float]]):
    #     donate = [False] * 5
    #     h, donate[0] = hidden.val, True
        
    #     if k == self.policy.num_gpu_batches - 1:
    #         (w_gate, donate[1]), (w_up, donate[2]), (w_down, donate[3]), (w_norm, donate[4]), = weight_read_buf.pop()
    #     else:
    #         (w_gate, _), (w_up, _), (w_down, _), (w_norm, _), = weight_read_buf.val
            
    #     h = self.computation.mlp(self.compute_device, h, w_gate, w_up, w_down, w_norm, self.config, donate)
    #     hidden.val = h


class LlamaTransformerLayer(BaseTransformerLayer):
    """Llama 的 Transformer 层。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map: dict, 
                 computation:LlamaModelComputation,  
                 freqs_cis_table: torch.Tensor):
        super().__init__(config, env, policy, weight_map)
        self.computation = computation
        self.attention = LlamaSelfAttention(config, env, policy, weight_map['attention'], self.computation)
        self.mlp = LlamaMLP(config, env, policy, weight_map['mlp'], self.computation)

        self.freqs_cis_table = freqs_cis_table
        
        # self.attention_norm_weight = None
        # self.ffn_norm_weight = None
    def init_weight(self, weight_home, path):
        home_attn, home_mlp, home_norm = ValueHolder(), ValueHolder(), ValueHolder()
        self.attention.init_weight(home_attn, path)
        self.mlp.init_weight(home_mlp, path)
        
        p = Path(path)
        h = self.config.hidden_size
        spec_attn_norm = [((h,), "atten_norm", p / self.weight_map["attention"]["norm"][0], self.config.dtype)]
        spec_ffn_norm = [((h,), "ffn_norm", p / self.weight_map["mlp"]["norm"][0], self.config.dtype)]
        norm_weights = (
            init_weight_list(spec_attn_norm, self.policy, self.env)[0],
            init_weight_list(spec_ffn_norm, self.policy, self.env)[0]
        )
        home_norm.store(norm_weights)
        weight_home.store((home_attn, home_mlp, home_norm))
    
    def load_weight(self, weight_home:ValueHolder, weight_read_buf:ValueHolder, k:List[Union[int, float]]):
        read_buf_attn, read_buf_mlp, read_buf_norm = ValueHolder(), ValueHolder(), ValueHolder()
        home_attn, home_mlp, home_norm = weight_home.val
        
        self.attention.load_weight(home_attn, read_buf_attn, k)
        self.mlp.load_weight(home_mlp, read_buf_mlp, k)
        
        if k == 0:
            # weight_read_buf.store((read_buf_attn, read_buf_mlp))
            w_attn_norm, w_ffn_norm = home_norm.val
            dst = self.compute_device
            read_buf_norm.store((w_attn_norm.smart_copy(dst), w_ffn_norm.smart_copy(dst)))
            weight_read_buf.store((read_buf_attn, read_buf_mlp, read_buf_norm))
    
    def init_cache_one_gpu_batch(self, cache_home:ValueHolder):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, 
                   cache_home:ValueHolder, 
                   cache_read_buf:ValueHolder, 
                   i:List[Union[int, float]]):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, 
                    cache_home:ValueHolder, 
                    cache_write_buf:ValueHolder, 
                    i:List[Union[int, float]]):
        self.attention.store_cache(cache_home, cache_write_buf, i)


    def set_task(self, task):
        self.task = task
    
    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:List[Union[int, float]], 
                k:List[Union[int, float]]):
        # TODO: 实现 Llama Transformer 层的完整前向逻辑
        # 逻辑: residual = hidden
        # hidden = self.input_layernorm(hidden, ...)
        # hidden = self.attention(hidden, ...)
        # hidden = residual + hidden
        # residual = hidden
        # hidden = self.post_attention_layernorm(hidden, ...)
        # hidden = self.mlp(hidden, ...)
        # hidden = residual + hidden
        is_prefill = (i == 0)
        donate = [False] * 2
        start_pos = 0 if is_prefill else self.task.prompt_len + i - 1
        seqlen = self.task.prompt_len if is_prefill else 1
        freqs_cis = self.freqs_cis_table[start_pos : start_pos + seqlen]
        
        h, residual = hidden.val.data, hidden.val.data
        if k == self.policy.num_gpu_batches - 1:
            buf_attn, buf_mlp, buf_norm = weight_read_buf.pop()
            (w_attn_norm, donate[0]), (w_ffn_norm, donate[1]), = buf_norm.val
        else:
            buf_attn, buf_mlp, buf_norm = weight_read_buf.val
            (w_attn_norm, _), (w_ffn_norm, _), = buf_norm.val
        
        # w_attn_norm, w_ffn_norm = buf_norm.val
        h_norm = self.computation.rms_norm(h, w_attn_norm.data)
        
        causal_mask = None
        if is_prefill:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
            causal_mask = torch.triu(mask, diagonal=1)
        
        past_kv = cache_read_buf.pop() if not is_prefill else None
        
        attn_output, k_new, v_new = self.attention.forward(
            TorchTensor.create_from_torch(h_norm, self.compute_device), causal_mask, 
            buf_attn.val, freqs_cis, start_pos, is_prefill, past_kv
        )
        if not is_prefill:
            cache_write_buf.store((k_new, v_new))
        h = residual + attn_output.data
        
        residual = h
        h_norm = self.computation.rms_norm(h, w_ffn_norm.data)
        mlp_output = self.mlp.forward(TorchTensor.create_from_torch(h_norm, self.compute_device), buf_mlp.val)
        h = residual + mlp_output.data
        
        hidden.val.data = h

        # if k == self.policy.num_gpu_batches - 1:
        #     read_buf_attn, read_buf_mlp = weight_read_buf.pop()
        # else:
        #     read_buf_attn, read_buf_mlp = weight_read_buf.val

        # self.attention.forward(hidden, cache_read_buf, read_buf_attn, attention_mask,
        #                        cache_write_buf, i, k)
        # self.mlp.forward(hidden, None, read_buf_mlp, attention_mask, None, i, k)


class LlamaModel(BaseModel):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 path:str):
        super().__init__(config, env, policy, weight_map=weight_map, path=path) 

        self.computation = LlamaModelComputation()

        s = getattr(self.config, "max_seq_len", None) or getattr(self.config, "max_position_embeddings", None)
        head_dim = config.hidden_size // config.n_head
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float().cuda() / head_dim))
        self.freqs_cis = self.computation.precompute_freqs_cis(head_dim, s * 2, inv_freq)

        self.layers.append(LlamaInputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))
        for layer_id in range(self.config.num_hidden_layers):
            self.layers.append(LlamaTransformerLayer(
                                                    self.config, self.env, self.policy, 
                                                    self.weight_map['layers'][layer_id], self.computation, 
                                                    self.freqs_cis))
        
        self.layers.append(LlamaRMSNorm(self.config, self.env, self.policy, self.weight_map, self.computation))
        self.layers.append(LlamaOutputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))


        self.init_cache_area() # 初始化用于保存权重 激活 cache 的内存区域

        self.task = None

        # Precompute rotary embedding frequencies
        # self.compute_device.rotary_emb_cis = self.computation.precompute_freqs_cis(
        #     self.config.hidden_size // self.config.n_head,
        #     s * 2,
        #     1.0 / (10000.0 ** (torch.arange(0, self.config.hidden_size // self.config.n_head, 2).float() / (self.config.hidden_size // self.config.n_head))).cuda()
        # )
        self.init_all_weights(flexgen_weight_path=self.path)

    def init_all_weights(self, flexgen_weight_path):
        # self.weight_home = array_1d(self.num_layers, ValueHolder)
        for layer_id, layer in enumerate(self.layers):
            layer.init_weight(self.weight_home[layer_id], flexgen_weight_path)