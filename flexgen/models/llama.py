from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Union, List, Tuple, Optional
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
    此类封装了所有原先在 pytorch_backend.py 中为 Llama 模型定义的计算逻辑。
    """
    
    def rms_norm(self, x: torch.Tensor, norm_weight: torch.Tensor, eps: float = 1e-5):
        """RMS Normalization - LLaMA使用RMSNorm而不是LayerNorm"""
        def _norm(x, eps):
            variance = x.pow(2).mean(-1, keepdim=True)
            # 移除 clamp 操作
            rsqrt_var = torch.rsqrt(variance + eps)
            return x * rsqrt_var
            
        output = _norm(x.float(), eps)
        result = output * norm_weight.float()
        return result.type_as(x)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """预计算RoPE的频率张量"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        """为RoPE重塑频率张量"""
        ndim = x.ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
        """
        应用旋转位置编码(RoPE) - 采用与Hugging Face transformers库更接近的实现方式。
        这种方式不直接使用torch.complex，而是通过手动旋转一半的维度来实现，
        这可能具有更好的数值稳定性。
        """
        def rotate_half(x):
            """旋转输入张量的一半隐藏维度。"""
            # 将最后一个维度切成两半
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            # 将后半部分取反，然后与前半部分拼接
            return torch.cat((-x2, x1), dim=-1)

        # 从复数形式的 freqs_cis 中提取 cos 和 sin
        # freqs_cis 的形状是 [seq_len, head_dim // 2]
        # 我们需要将其 reshape 以便和 xq, xk 进行广播
        cos = freqs_cis.real.to(xq.dtype)
        sin = freqs_cis.imag.to(xq.dtype)
        
        # unsqueeze(0) for batch dim, unsqueeze(2) for n_head dim
        # 最终形状变为 [1, seq_len, 1, head_dim // 2] 以匹配 (b, s, h, d)
        # 但是 LLaMA 的 RoPE 是在 head_dim 级别操作的，所以需要重复
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # 广播 cos 和 sin 到完整的 head_dim
        # [1, s, 1, d/2] -> [1, s, 1, d]
        cos = cos.repeat(1, 1, 1, 2)
        sin = sin.repeat(1, 1, 1, 2)

        # 应用旋转编码
        xq_out = (xq * cos) + (rotate_half(xq) * sin)
        xk_out = (xk * cos) + (rotate_half(xk) * sin)
        
        return xq_out, xk_out

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """为Grouped Query Attention重复key和value"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def input_embed(self, compute_device, inputs, w_token, pad_token_id, donate):
        """输入嵌入层 - LLaMA只有token embedding，没有position embedding"""
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        if donate[0]: inputs.delete()
        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        return TorchTensor.create_from_torch(token_embed, compute_device)

    def output_embed(self, compute_device, inputs, w_ln, w_token, donate,
                     do_sample, temperature):
        """输出嵌入层 - 使用RMSNorm"""
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape
        
        # 保存输入隐藏状态数据
        if hasattr(self, 'debug_data'):
            self.debug_data['input_hidden_state'] = {
                'shape': list(inputs.data.shape),
                'range': [inputs.data.min().item(), inputs.data.max().item()],
                'mean': inputs.data.mean().item(),
                'std': inputs.data.std().item(),
                'tensor': inputs.data.cpu().numpy()
            }
        
        # RMSNorm instead of LayerNorm
        hidden = self.rms_norm(inputs.data, w_ln.data)

        
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        
        
        return TorchTensor.create_from_torch(ids, compute_device), TorchTensor.create_from_torch(logits, compute_device)

    
    def mha(self, compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
        n_head, n_kv_heads, freqs_cis, donate, compress_cache, comp_config):
        """
        mha 的最终调试版本，会打印每一步的中间结果。
        """

        b, s, h = inputs.shape
        head_dim = h // n_head
        

        # 1. 前置归一化
        residual = inputs.data
        hidden = self.rms_norm(inputs.data, w_ln.data)
       

        # 2. QKV 线性投射
        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
       

        # 3. Reshape 和 Transpose
        query_states = q.view(b, s, n_head, head_dim).transpose(1, 2)
        key_states = k.view(b, s, n_kv_heads, head_dim).transpose(1, 2)
        value_states = v.view(b, s, n_kv_heads, head_dim).transpose(1, 2)

        # 4. RoPE
        cos_half = freqs_cis.real.to(query_states.dtype); sin_half = freqs_cis.imag.to(query_states.dtype)
        cos = torch.cat((cos_half, cos_half), dim=-1); sin = torch.cat((sin_half, sin_half), dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0); sin = sin.unsqueeze(0).unsqueeze(0)
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        # 5. GQA
        k_for_cache = key_states.transpose(1,2).contiguous()
        v_for_cache = value_states.transpose(1,2).contiguous()
        def repeat_kv_official(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            batch, num_key_value_heads, slen, head_dim_inner = hidden_states.shape
            if n_rep == 1: return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim_inner)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim_inner)
        key_states = repeat_kv_official(key_states, n_head // n_kv_heads)
        value_states = repeat_kv_official(value_states, n_head // n_kv_heads)

        # 6. 注意力得分
        scaling = head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        # # 7. 遮罩
        mask_fill_value = torch.finfo(torch.float16).min
    
        if s > 1:
            causal_mask = torch.triu(torch.full((s, s), mask_fill_value, device=attn_weights.device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + causal_mask
        if attention_mask is not None:
            padding_mask = ~(attention_mask.data.bool()).view(b, 1, 1, s)
            attn_weights = attn_weights.masked_fill(padding_mask, mask_fill_value)

        # 8. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 9. 应用于V
        attn_output = torch.matmul(attn_weights, value_states)

        # 10. 输出 reshape 和投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(b, s, h)
        value = F.linear(attn_output, w_out.data) + residual

        if donate[0]: inputs.delete(); 
        if donate[1]: attention_mask.delete()
        k = k_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)
        v = v_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)

        if compress_cache:
            k = compute_device.compressed_device.compress(k, comp_config)
            v = compute_device.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, compute_device)
            v = TorchTensor.create_from_torch(v, compute_device)
        return TorchTensor.create_from_torch(value, compute_device), k, v

    def mha_gen(self, compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
            n_head, n_kv_heads, k_cache, v_cache, freqs_cis, donate,
            attn_sparsity, compress_cache, comp_config):
        """
        Multi-head attention (decoding phase) - Final corrected version for K/V cache shaping.
        """
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q, w_k, w_v, w_out = [w.device.decompress(w) for w in [w_q, w_k, w_v, w_out]]

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head

        # 1. 前置归一化
        residual = inputs.data
        hidden = self.rms_norm(inputs.data, w_ln.data)

        # 2. QKV 线性投射
        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)

        # 3. 重塑和转置
        query_states = q.view(b, tgt_s, n_head, head_dim).transpose(1, 2)
        key_states = k.view(b, tgt_s, n_kv_heads, head_dim).transpose(1, 2)
        value_states = v.view(b, tgt_s, n_kv_heads, head_dim).transpose(1, 2)

        # 4. RoPE
        cos_half = freqs_cis.real.to(query_states.dtype)
        sin_half = freqs_cis.imag.to(query_states.dtype)
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        
        # 5. 准备 k_new 和 v_new 用于 cache
        k_new_for_cache = key_states.permute(2, 0, 1, 3).contiguous().reshape(-1, b * n_kv_heads, head_dim)
        v_new_for_cache = value_states.permute(2, 0, 1, 3).contiguous().reshape(-1, b * n_kv_heads, head_dim)

        if isinstance(k_cache, TorchTensor):
            if compress_cache:
                k_cached = k_cache.device.decompress(k_cache)[:src_s-tgt_s]
                v_cached = v_cache.device.decompress(v_cache)[:src_s-tgt_s]
            else:
                k_cached = k_cache.data[:src_s-tgt_s]
                v_cached = v_cache.data[:src_s-tgt_s]
            
            k_all = torch.cat([k_cached, k_new_for_cache], dim=0) # shape: (src_s, b * n_kv_heads, head_dim)
            v_all = torch.cat([v_cached, v_new_for_cache], dim=0)
        else:
            k_all = k_new_for_cache # shape: (tgt_s, b * n_kv_heads, head_dim)
            v_all = v_new_for_cache # shape: (tgt_s, b * n_kv_heads, head_dim)
        
        # 直接从 k_all 张量获取其真实的序列长度
        actual_seq_len = k_all.shape[0]
        # 使用真实长度进行 view 操作，并为保证内存连续性添加 .contiguous()
        key_states = k_all.permute(1, 0, 2).contiguous().view(b, n_kv_heads, actual_seq_len, head_dim)
        value_states = v_all.permute(1, 0, 2).contiguous().view(b, n_kv_heads, actual_seq_len, head_dim)

        def repeat_kv_official(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            batch, num_key_value_heads, slen, head_dim_inner = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim_inner)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim_inner)

        key_states = repeat_kv_official(key_states, n_head // n_kv_heads)
        value_states = repeat_kv_official(value_states, n_head // n_kv_heads)


        # 6. 注意力得分
        scaling = head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        
        # 7. 应用注意力遮罩
        mask_fill_value = torch.finfo(attn_weights.dtype).min
        if attention_mask is not None:
            padding_mask = ~(attention_mask.data.bool()).view(b, 1, 1, actual_seq_len)
            attn_weights = attn_weights.masked_fill(padding_mask, mask_fill_value)

        # 8. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 9. 应用于 V
        attn_output = torch.matmul(attn_weights, value_states)

        # 10. 输出 reshape 和投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(b, tgt_s, h)
        value = F.linear(attn_output, w_out.data) + residual

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            k_new = compute_device.compressed_device.compress(k_new_for_cache, comp_config)
            v_new = compute_device.compressed_device.compress(v_new_for_cache, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new_for_cache, compute_device)
            v_new = TorchTensor.create_from_torch(v_new_for_cache, compute_device)

        return TorchTensor.create_from_torch(value, compute_device), k_new, v_new

    def mlp(self, compute_device, inputs, w_gate, w_up, w_down, w_ln, donate):
        """MLP with SwiGLU activation - LLaMA特有的激活函数"""
        # decompress weights
        if w_gate.device.device_type == DeviceType.COMPRESSED:
            w_gate = w_gate.device.decompress(w_gate)
            w_up = w_up.device.decompress(w_up)
            w_down = w_down.device.decompress(w_down)

        b, s, h = inputs.shape
        
        # Pre-normalization with RMSNorm
        residual = inputs.data
        out = self.rms_norm(inputs.data, w_ln.data)

        
        gate = F.linear(out, w_gate.data)
        up = F.linear(out, w_up.data)       
        silu_gate = F.silu(gate)  # SiLU activation
        
        intermediate = silu_gate * up  # Element-wise multiplication
        
        # Down projection
        out = F.linear(intermediate, w_down.data)
        
        out = out + residual  # Residual connection

        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, compute_device)

# Llama 2 模型与 OPT 的主要区别:
# 1. 位置编码: 使用旋转位置编码 (Rotary Positional Embedding, RoPE)，在 Attention 层中应用，而不是独立的输入层。
# 2. 归一化层: 使用 RMSNorm (Root Mean Square Normalization)，而不是 LayerNorm。
# 3. MLP/FFN 层: 使用带 SiLU 激活函数的 SwiGLU，而不是带 ReLU 的标准 MLP。
# 4. 注意力机制: 可能使用分组查询注意力 (Grouped-Query Attention, GQA) 来优化推理。

class LLaMAInputEmbed(BaseModelLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LlamaModelComputation'):
        super().__init__(config, env, policy, weight_map=weight_map) 
        self.computation = computation  
        
    def init_weight(self, 
                    weight_home: ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        p = Path(converted_path)
        
        # LLaMA 只有词嵌入，没有位置嵌入（使用RoPE）
        weight_specs = [
            # ((v, h), "embed_tokens", p / self.weight_map["embed_tokens"][0], self.config.dtype),
            ((v, h), "embed_tokens", p / self.weight_map["embed_tokens"][0], self.weight_map["embed_tokens"][1]),
        ]
        
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    k: List[Union[int, float]]):
        w_token, = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst),))

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64
    
    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):

        # Compute input embedding
        donate = [False] * 2
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            (w_token, donate[1]), = weight_read_buf.pop()
        else:
            (w_token, _), = weight_read_buf.val

        h = self.computation.input_embed(self.compute_device, h,
            w_token, getattr(self.config, "pad_token_id", self.config.bos_token_id), donate)
        hidden.val = h



class LLaMAOutputEmbed(BaseModelLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LlamaModelComputation'):
        super().__init__(config, env, policy, weight_map=weight_map)  
        self.computation = computation 
        
    def init_weight(self, 
                    weight_home: ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        v = self.config.vocab_size
        p = Path(converted_path)
        
        # 输出层权重 - 使用RMSNorm和lm_head
        weight_specs = [
            # ((h,), "final_norm_weight", p / "model.norm.weight", self.config.dtype),
            # ((v, h), "lm_head", p / self.weight_map['lm_head'][0], self.config.dtype),  # 使用权重映射
            ((h,), "final_norm_weight", p / self.weight_map['final_norm'][0], self.weight_map['final_norm'][1]),
            ((v, h), "lm_head", p / self.weight_map['lm_head'][0], self.weight_map['lm_head'][1]),  # 使用权重映射
        ]

        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    k: List[Union[int, float]]):
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute_device
            weight_read_buf.store((w_ln.smart_copy(dst2), w_token.smart_copy(dst1)))
    
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        h, logits = self.computation.output_embed(self.compute_device, h, w_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        if self.task.logits:
            hidden.val = [h, logits]
        else:
            hidden.val = h


class LLaMASelfAttention(BaseModelLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LlamaModelComputation', 
                 ):
        super().__init__(config, env, policy, weight_map=weight_map) 
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)  
        self.computation = computation 

        # LLaMA specific parameters
        self.n_head = config.n_head
        self.n_kv_heads = getattr(config, 'num_key_value_heads', self.n_head)
        self.head_dim = config.hidden_size // self.n_head
        
        # Precompute RoPE frequencies
        max_seq_len = getattr(config, "max_seq_len", None) or getattr(config, "max_position_embeddings", 2048)
        self.freqs_cis = self.computation.precompute_freqs_cis(
            self.head_dim, max_seq_len * 2
        )

    def init_weight(self, 
                    weight_home: ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        kv_dim = self.n_kv_heads * self.head_dim
        p = Path(converted_path)
        
        weight_specs = [
            # Query projection
            ((h, h), "attn_q_proj_weight", p / self.weight_map["q_proj"][0], self.weight_map["q_proj"][1]),
            # Key projection (potentially smaller for GQA)
            ((kv_dim, h), "attn_k_proj_weight", p / self.weight_map["k_proj"][0], self.weight_map["k_proj"][1]),
            # Value projection (potentially smaller for GQA)
            ((kv_dim, h), "attn_v_proj_weight", p / self.weight_map["v_proj"][0], self.weight_map["v_proj"][1]),
            # Output projection
            ((h, h), "attn_o_proj_weight", p / self.weight_map["o_proj"][0], self.weight_map["o_proj"][1]),
            # RMSNorm weight (no bias)
            ((h,), "attn_norm_weight", p / self.weight_map["norm"][0], self.weight_map["norm"][1]),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    k: List[Union[int, float]]):
        w_q, w_k, w_v, w_out, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute_device
            weight_read_buf.store((
                w_q.smart_copy(dst1), w_k.smart_copy(dst1),
                w_v.smart_copy(dst1), w_out.smart_copy(dst1),
                w_ln.smart_copy(dst2)))
    
    def init_cache_one_gpu_batch(self, cache_home: ValueHolder):
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
                   cache_home: ValueHolder, 
                   cache_read_buf: ValueHolder, 
                   i: List[Union[int, float]]):
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
            indices = (slice(0, self.task.prompt_len + i - 1),
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
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

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
                    cache_home: ValueHolder, 
                    cache_write_buf: ValueHolder, 
                    i: List[Union[int, float]]):
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
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        
        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            ((w_q, donate[2]), (w_k, donate[3]), (w_v, donate[4]), 
             (w_out, donate[5]), (w_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((w_q, _), (w_k, _), (w_v, _), (w_out, _), (w_ln, _)) = weight_read_buf.val

        # Get RoPE frequencies for current position
        seq_len = h.shape[1]
        start_pos = 0 if i == 0 else self.task.prompt_len + i - 1
        # 确保freqs_cis在正确的设备上
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        if hasattr(h, 'device') and hasattr(h.device, 'dev'):
            freqs_cis = freqs_cis.to(h.device.dev)
        else:
            freqs_cis = freqs_cis.to(h.device)

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute_device)
            h, new_k_cache, new_v_cache = self.computation.mha(
                self.compute_device, h, mask, w_q, w_k, w_v, w_out, w_ln, 
                self.n_head, self.n_kv_heads, freqs_cis, donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[7]), (v_cache, donate[8]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.computation.mha_gen(
                self.compute_device, h, mask, w_q, w_k, w_v, w_out, w_ln,
                self.n_head, self.n_kv_heads, k_cache, v_cache, freqs_cis, donate,
                self.policy.attn_sparsity, self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class LLaMAMLP(BaseModelLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LlamaModelComputation'):
        super().__init__(config, env, policy, weight_map=weight_map) 
        self.computation = computation

    def init_weight(self, 
                    weight_home: ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        intermediate_size = getattr(self.config, 'intermediate_size', 4 * h)
        p = Path(converted_path)
        
        # LLaMA MLP has gate, up, down projections for SwiGLU
        weight_specs = [
            ((intermediate_size, h), "mlp_gate_proj_weight", p / self.weight_map["gate_proj"][0], self.weight_map["gate_proj"][1]),
            ((intermediate_size, h), "mlp_up_proj_weight", p / self.weight_map["up_proj"][0], self.weight_map["up_proj"][1]),
            ((h, intermediate_size), "mlp_down_proj_weight", p / self.weight_map["down_proj"][0], self.weight_map["down_proj"][1]),
            ((h,), "mlp_norm_weight", p / self.weight_map["norm"][0], self.weight_map["norm"][1]),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    k: List[Union[int, float]]):
        w_gate, w_up, w_down, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute_device
            weight_read_buf.store((
                w_gate.smart_copy(dst1), w_up.smart_copy(dst1),
                w_down.smart_copy(dst1), w_ln.smart_copy(dst2)))
    
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), np.float16
    
    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            ((w_gate, donate[1]), (w_up, donate[2]), 
             (w_down, donate[3]), (w_ln, donate[4])) = weight_read_buf.pop()
        else:
            ((w_gate, _), (w_up, _), (w_down, _), (w_ln, _)) = weight_read_buf.val

        h = self.computation.mlp(self.compute_device, h, w_gate, w_up, w_down, w_ln, donate)
        hidden.val = h


class LLaMATransformerLayer(BaseTransformerLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LlamaModelComputation',):
        super().__init__(config, env, policy, weight_map)
        self.attention = LLaMASelfAttention(config=config, env=env, policy=policy, 
                                           weight_map=weight_map['attention'], computation=computation)
        self.mlp = LLaMAMLP(config=config, env=env, policy=policy, 
                           weight_map=weight_map['mlp'], computation=computation)

    def init_weight(self, 
                    weight_home, 
                    path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))
    
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    k: List[Union[int, float]]):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home: ValueHolder):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, 
                   cache_home: ValueHolder, 
                   cache_read_buf: ValueHolder, 
                   i: List[Union[int, float]]):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, 
                    cache_home: ValueHolder, 
                    cache_write_buf: ValueHolder, 
                    i: List[Union[int, float]]):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k)
        
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)



class LLaMAModel(BaseModel):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 path:str):
        super().__init__(config, env, policy, weight_map=weight_map, path=path) 

        self.computation = LlamaModelComputation()
        
        # 初始化调试数据
        self.debug_data = {}
        self.computation.debug_data = self.debug_data
        self.policy.sep_layer = False
        # InputEmbed, TransformerLayer, OutputEmbed = get_model_architecture(self.config.model_type)
        self.layers.append(LLaMAInputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))
        for layer_id in range(self.config.num_hidden_layers):
            if self.policy.sep_layer:
                self.layers.append(LLaMASelfAttention(self.config, self.env, self.policy, self.weight_map['layers'][layer_id]['attention'], self.computation))
                self.layers.append(LLaMAMLP(self.config, self.env, self.policy, self.weight_map['layers'][layer_id]['mlp'], self.computation))
            else:
                self.layers.append(LLaMATransformerLayer(self.config, self.env, self.policy, self.weight_map['layers'][layer_id], self.computation))
        self.layers.append(LLaMAOutputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))


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
        
        self.init_all_weights(flexgen_weight_path=self.path)

    def init_all_weights(self, flexgen_weight_path):
        for layer_id, layer in enumerate(self.layers):
            layer.init_weight(self.weight_home[layer_id], flexgen_weight_path)
    