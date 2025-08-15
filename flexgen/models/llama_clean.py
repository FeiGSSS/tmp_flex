from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Union, List, Optional

from flexgen.models.base import BaseModelLayer, BaseTransformerLayer, BaseModel
from flexgen.utils import (ValueHolder, 
                           array_1d, array_2d, array_3d, array_4d)
from flexgen.models.utils import (ExecutionEnv, Task, Policy, 
                                  init_weight_list, get_weight_tuple)
from flexgen.models.config import FlexModelConfig
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink, TorchNuma, TorchTensor, 
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)

fix_recursive_import()

import pickle as pkl
import os
from collections import defaultdict

class DebugLogger:
    """
    一个用于记录和保存调试张量的类。
    它将数据缓存在内存中，并在最后一次性写入磁盘，以提高效率。
    """
    def __init__(self, output_dir: str, filename: str = "flexgen_debug_data.pkl"):
        self.output_path = os.path.join(output_dir, filename)
        # 使用defaultdict简化嵌套字典的创建
        self.data_cache = defaultdict(lambda: defaultdict(dict))
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 在开始时删除旧文件，确保每次运行都是全新的记录
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        
        print(f"DebugLogger initialized. Outputs will be saved to: {self.output_path}")

    def record(self, tensor_data: torch.Tensor, gen_token_step: int, layer_id: int, layer_name: str):
        """
        记录一个张量到内存缓存中。
        
        Args:
            tensor_data (torch.Tensor): 需要保存的张量 (会自动转到CPU).
            gen_token_step (int): 当前是第几个生成步骤 (e.g., prefill时为0).
            layer_id (int): 当前层的ID.
            layer_name (str): 当前层的名称 (e.g., 'input_embed', 'transformer_output').
        """
        # 使用setdefault来优雅地处理嵌套结构
        layer_data = self.data_cache[gen_token_step].setdefault(layer_id, {})
        
        # 存储张量数据
        layer_data[layer_name] = {
            "full_tensor": tensor_data.cpu()
        }

    def save_to_disk(self):
        """
        将缓存中的所有数据一次性写入pickle文件。
        """
        print(f"Saving all debug data to {self.output_path}...")
        # 将defaultdict转换为普通dict以便更好地序列化和读取
        final_data = {k: dict(v) for k, v in self.data_cache.items()}

        with open(self.output_path, "wb") as f:
            pkl.dump(final_data, f)
        print("Debug data saved successfully.")

# 创建一个全局的logger实例，方便在项目各处调用
# 您可以根据需要修改这里的路径
FLEXGEN_DEBUG_LOGGER = DebugLogger(output_dir="/home/xu/code/FlexGen/tmp_flex/debug_results")

# 在文件开头添加调试函数
def print_stats(tensor, name):
    """打印张量的统计信息，格式与test_final_verify_and_trace.py一致"""
    if isinstance(tensor, torch.Tensor):
        print(f"\n>> [FlexGen] {name}:")
        print(f"   - Shape: {tensor.shape}")
        print(f"   - Dtype: {tensor.dtype}")
        print(f"   - Mean:  {tensor.float().mean().item():.6f}")
        print(f"   - Max:   {tensor.float().max().item():.6f}")
        print(f"   - Min:   {tensor.float().min().item():.6f}")
        print(f"   - Std:   {tensor.float().std().item():.6f}")
    elif hasattr(tensor, 'data') and isinstance(tensor.data, torch.Tensor):
        # 处理TorchTensor类型
        print(f"\n>> [FlexGen] {name}:")
        print(f"   - Shape: {tensor.data.shape}")
        print(f"   - Dtype: {tensor.data.dtype}")
        print(f"   - Mean:  {tensor.data.float().mean().item():.6f}")
        print(f"   - Max:   {tensor.data.float().max().item():.6f}")
        print(f"   - Min:   {tensor.data.float().min().item():.6f}")
        print(f"   - Std:   {tensor.data.float().std().item():.6f}")

def debug_token_generation(logits, tokenizer, top_k=10):
    """调试token生成过程"""
    print(f"DEBUG: Logits shape: {logits.shape}")
    print(f"DEBUG: Logits dtype: {logits.dtype}")
    
    # 检查logits是否异常
    if torch.isinf(logits).any() or torch.isnan(logits).any():
        print(f"ERROR: Logits contains inf/nan!")
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # 显示前k个最高的logits值和对应的token
    top_logits, top_indices = torch.topk(logits[0], top_k)
    print(f"DEBUG: Top {top_k} logits: {top_logits.tolist()}")
    print(f"DEBUG: Top {top_k} indices: {top_indices.tolist()}")
    
    # 尝试解码top tokens
    try:
        for i, token_id in enumerate(top_indices.tolist()):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"DEBUG: Token {i+1}: ID={token_id}, Text='{token_text}'")
    except Exception as e:
        print(f"DEBUG: Failed to decode tokens: {e}")
    
    return top_indices[0].item()  # 返回最高概率的token ID


class LLaMAModelComputation:
    """
    此类封装了所有原先在 pytorch_backend.py 中为 LLaMA 模型定义的计算逻辑。
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
        # print(f"DEBUG: token_ids shape and dtype: {token_ids.shape}, {token_ids.dtype}")
        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        # print(f"DEBUG: token_embed shape and dtype: {token_embed.shape}, {token_embed.dtype}")
        # exit()
        return TorchTensor.create_from_torch(token_embed, compute_device)

    def output_embed(self, compute_device, inputs, w_ln, w_token, donate,
                     do_sample, temperature):
        """输出嵌入层 - 使用RMSNorm"""
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape
        
        # 调试隐藏状态
        # print(f"DEBUG: 输入隐藏状态形状: {inputs.data.shape}")
        # print(f"DEBUG: 输入隐藏状态范围: [{inputs.data.min().item():.4f}, {inputs.data.max().item():.4f}]")
        # print(f"DEBUG: 输入隐藏状态均值: {inputs.data.mean().item():.4f}, 标准差: {inputs.data.std().item():.4f}")
        
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
        
        # 调试RMSNorm后的隐藏状态
        # print(f"DEBUG: RMSNorm后隐藏状态范围: [{hidden.min().item():.4f}, {hidden.max().item():.4f}]")
        # print(f"DEBUG: RMSNorm后隐藏状态均值: {hidden.mean().item():.4f}, 标准差: {hidden.std().item():.4f}")
        
        # 保存RMSNorm后的隐藏状态数据
        if hasattr(self, 'debug_data'):
            self.debug_data['normalized_hidden_state'] = {
                'range': [hidden.min().item(), hidden.max().item()],
                'mean': hidden.mean().item(),
                'std': hidden.std().item(),
                'tensor': hidden.cpu().numpy()
            }
        
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        # 详细的调试信息
        # print(f"DEBUG: Last token logits shape: {last_token_logits.shape}")
        # print(f"DEBUG: do_sample={do_sample}, temperature={temperature}")
        # print(f"DEBUG: Logits range: [{last_token_logits.min().item():.4f}, {last_token_logits.max().item():.4f}]")
        # print(f"DEBUG: Logits mean: {last_token_logits.mean().item():.4f}, std: {last_token_logits.std().item():.4f}")
        
        # 保存logits数据
        if hasattr(self, 'debug_data'):
            self.debug_data['logits'] = {
                'shape': list(logits.shape),
                'range': [logits.min().item(), logits.max().item()],
                'mean': logits.mean().item(),
                'std': logits.std().item(),
                'tensor': logits.cpu().numpy(),
                'last_token_logits': {
                    'shape': list(last_token_logits.shape),
                    'range': [last_token_logits.min().item(), last_token_logits.max().item()],
                    'mean': last_token_logits.mean().item(),
                    'std': last_token_logits.std().item(),
                    'tensor': last_token_logits.cpu().numpy()
                }
            }
        
        # 检查logits是否异常
        if torch.isinf(last_token_logits).any() or torch.isnan(last_token_logits).any():
            print(f"ERROR: Logits contains inf/nan!")
            last_token_logits = torch.nan_to_num(last_token_logits, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # 显示前10个最高的logits值和对应的token ID
        top_logits, top_indices = torch.topk(last_token_logits[0], 10)
        # print(f"DEBUG: Top 10 logits: {top_logits.tolist()}")
        # print(f"DEBUG: Top 10 indices: {top_indices.tolist()}")
        
        # 保存top-k数据
        if hasattr(self, 'debug_data'):
            self.debug_data['top_k'] = {
                'logits': top_logits.cpu().numpy(),
                'indices': top_indices.cpu().numpy()
            }

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        
        # print(f"DEBUG: Generated token IDs: {ids.cpu().numpy()}")
        # print(f"DEBUG: Generated token ID range: [{ids.min().item()}, {ids.max().item()}]")
        
        # 检查生成的token ID是否在合理范围内
        vocab_size = w_token.data.shape[0]
        if ids.max().item() >= vocab_size:
            print(f"ERROR: Generated token ID {ids.max().item()} >= vocab_size {vocab_size}")
        if ids.min().item() < 0:
            print(f"ERROR: Generated token ID {ids.min().item()} < 0")
        
        # 检查是否总是生成相同的token（循环检测）
        if hasattr(self, '_last_generated_id'):
            if torch.equal(ids, self._last_generated_id):
                print(f"WARNING: Generated same token as last time!")
        self._last_generated_id = ids.clone()
        
        # print("="*60)
        
        return TorchTensor.create_from_torch(ids, compute_device), TorchTensor.create_from_torch(logits, compute_device)

    # def mha_old(self, compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln, 
    #         n_head, n_kv_heads, freqs_cis, donate, compress_cache, comp_config):
    #     """Multi-head attention (prefill phase) with RoPE and GQA"""
    #     # decompress weights
    #     if w_q.device.device_type == DeviceType.COMPRESSED:
    #         w_q = w_q.device.decompress(w_q)
    #         w_k = w_k.device.decompress(w_k)
    #         w_v = w_v.device.decompress(w_v)
    #         w_out = w_out.device.decompress(w_out)

    #     b, s, h = inputs.shape
    #     head_dim = h // n_head
    #     scaling = head_dim ** -0.5
        
    #     # Pre-normalization with RMSNorm
    #     residual = inputs.data
    #     hidden = self.rms_norm(inputs.data, w_ln.data)

    #     # Linear projections
    #     q = F.linear(hidden, w_q.data) * scaling
    #     k = F.linear(hidden, w_k.data)
    #     v = F.linear(hidden, w_v.data)
        
    #     # Reshape for multi-head attention
    #     q = q.view(b, s, n_head, head_dim)
    #     k = k.view(b, s, n_kv_heads, head_dim)
    #     v = v.view(b, s, n_kv_heads, head_dim)

    #     # Apply RoPE
    #     q, k = self.apply_rotary_emb(q, k, freqs_cis)

    #     # Save for cache before repeating
    #     k_for_cache = k
    #     v_for_cache = v
        
    #     # Repeat k,v for grouped query attention
    #     num_key_value_groups = n_head // n_kv_heads
    #     k = self.repeat_kv(k, num_key_value_groups)
    #     v = self.repeat_kv(v, num_key_value_groups)

    #     # Transpose for attention computation
    #     q = q.transpose(1, 2)  # (b, n_head, s, head_dim)
    #     k = k.transpose(1, 2)  # (b, n_head, s, head_dim)
    #     v = v.transpose(1, 2)  # (b, n_head, s, head_dim)

    #     # Compute attention scores
    #     scores = torch.matmul(q, k.transpose(2, 3))

    #     # Apply attention mask (combine causal mask with attention mask)
    #     if s > 1:
    #         # Causal mask
    #         causal_mask = torch.full((1, 1, s, s), float("-inf"), device=scores.device)
    #         causal_mask = torch.triu(causal_mask, diagonal=1)
    #         scores = scores + causal_mask
        
    #     # Apply attention mask if provided
    #     if attention_mask is not None:
    #         mask = attention_mask.data.view(b, 1, 1, s)
    #         scores = torch.where(mask, scores, -1e4)
        
    #     scores = F.softmax(scores.float(), dim=-1).type_as(q)
    #     output = torch.matmul(scores, v)

    #     # Reshape and apply output projection
    #     output = output.transpose(1, 2).contiguous().view(b, s, h)
    #     value = F.linear(output, w_out.data) + residual

    #     if donate[0]: inputs.delete()
    #     if donate[1]: attention_mask.delete()

    #     # Prepare cache tensors (s, b * n_kv_heads, head_dim)
    #     k = k_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)
    #     v = v_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)

    #     if compress_cache:
    #         k = compute_device.compressed_device.compress(k, comp_config)
    #         v = compute_device.compressed_device.compress(v, comp_config)
    #     else:
    #         k = TorchTensor.create_from_torch(k, compute_device)
    #         v = TorchTensor.create_from_torch(v, compute_device)

    #     return TorchTensor.create_from_torch(value, compute_device), k, v
    # def mha(self, compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
    #     n_head, n_kv_heads, freqs_cis, donate, compress_cache, comp_config):
    #     """
    #     Multi-head attention (prefill phase) - 已根据官方 modeling_llama.py 的逻辑重构
    #     此版本修正了 RoPE 维度并补全了 Causal Masking 逻辑。
    #     """
    #     if w_q.device.device_type == DeviceType.COMPRESSED:
    #         w_q, w_k, w_v, w_out = [w.device.decompress(w) for w in [w_q, w_k, w_v, w_out]]

    #     b, s, h = inputs.shape
    #     head_dim = h // n_head
        
    #     residual = inputs.data
    #     hidden = self.rms_norm(inputs.data, w_ln.data)

    #     q = F.linear(hidden, w_q.data)
    #     k = F.linear(hidden, w_k.data)
    #     v = F.linear(hidden, w_v.data)

    #     query_states = q.view(b, s, n_head, head_dim).transpose(1, 2)
    #     key_states = k.view(b, s, n_kv_heads, head_dim).transpose(1, 2)
    #     value_states = v.view(b, s, n_kv_heads, head_dim).transpose(1, 2)

    #     cos_half = freqs_cis.real.to(query_states.dtype)
    #     sin_half = freqs_cis.imag.to(query_states.dtype)
    #     cos = torch.cat((cos_half, cos_half), dim=-1)
    #     sin = torch.cat((sin_half, sin_half), dim=-1)
    #     cos = cos.unsqueeze(0).unsqueeze(0)
    #     sin = sin.unsqueeze(0).unsqueeze(0)

    #     def rotate_half(x):
    #         x1 = x[..., : x.shape[-1] // 2]
    #         x2 = x[..., x.shape[-1] // 2 :]
    #         return torch.cat((-x2, x1), dim=-1)

    #     query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    #     key_states = (key_states * cos) + (rotate_half(key_states) * sin)

    #     num_key_value_groups = n_head // n_kv_heads
        
    #     def repeat_kv_official(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    #         batch, num_key_value_heads, slen, head_dim_inner = hidden_states.shape
    #         if n_rep == 1:
    #             return hidden_states
    #         hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim_inner)
    #         return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim_inner)

    #     k_for_cache = key_states.transpose(1,2).contiguous()
    #     v_for_cache = value_states.transpose(1,2).contiguous()

    #     key_states = repeat_kv_official(key_states, num_key_value_groups)
    #     value_states = repeat_kv_official(value_states, num_key_value_groups)
        
    #     scaling = head_dim ** -0.5
    #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    #     # --- !!! 以下是针对本次问题的最终修正 !!! ---

    #     # 修正点：必须同时应用 Causal Mask 和 Padding Mask
    #     # `attention_mask` 是从FlexGen框架传入的，它只包含了padding信息
        
    #     # 1. 创建并应用因果遮罩 (Causal Mask)
    #     if s > 1:
    #         # 创建一个上三角矩阵，对角线以上的位置将被屏蔽
    #         causal_mask = torch.triu(torch.full((s, s), torch.finfo(attn_weights.dtype).min, device=attn_weights.device), diagonal=1)
    #         # 将其扩展到和 attn_weights 同样的维度 (b, n_head, s, s)
    #         causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    #         attn_weights = attn_weights + causal_mask

    #     # 2. 应用 padding mask (如果存在)
    #     #    FlexGen传入的 attention_mask 是 (b, s) 的布尔矩阵，True代表有效token
    #     if attention_mask is not None:
    #         # 将 padding mask 变形以进行广播
    #         # (b, s) -> (b, 1, 1, s)
    #         padding_mask = (attention_mask.data == 0).view(b, 1, 1, s)
    #         # 在 padding 的位置填充一个极小值
    #         attn_weights = attn_weights.masked_fill(padding_mask, torch.finfo(attn_weights.dtype).min)

    #     # ------------------- 修正结束 -------------------
        
    #     attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #     attn_output = torch.matmul(attn_weights, value_states)

    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.reshape(b, s, h)
        
    #     value = F.linear(attn_output, w_out.data) + residual

    #     if donate[0]: inputs.delete()
    #     if donate[1]: attention_mask.delete()

    #     k = k_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)
    #     v = v_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)

    #     if compress_cache:
    #         k = compute_device.compressed_device.compress(k, comp_config)
    #         v = compute_device.compressed_device.compress(v, comp_config)
    #     else:
    #         k = TorchTensor.create_from_torch(k, compute_device)
    #         v = TorchTensor.create_from_torch(v, compute_device)

    #     return TorchTensor.create_from_torch(value, compute_device), k, v
    def mha(self, compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
        n_head, n_kv_heads, freqs_cis, donate, compress_cache, comp_config):
        """
        mha 的最终调试版本，会打印每一步的中间结果。
        """

        b, s, h = inputs.shape
        head_dim = h // n_head
        
        # 0. 输入
        # print_stats(inputs, "0. Initial Hidden State")

        # 1. 前置归一化
        residual = inputs.data
        hidden = self.rms_norm(inputs.data, w_ln.data)
        # print_stats(hidden, "1. After RMSNorm")

        # 2. QKV 线性投射
        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # print_stats(q, "2a. Q after Linear")
        # print_stats(k, "2b. K after Linear")
        # print_stats(v, "2c. V after Linear")

        # 3. Reshape 和 Transpose
        query_states = q.view(b, s, n_head, head_dim).transpose(1, 2)
        key_states = k.view(b, s, n_kv_heads, head_dim).transpose(1, 2)
        value_states = v.view(b, s, n_kv_heads, head_dim).transpose(1, 2)
        # print_stats(query_states, "3. Q after Reshape & Transpose")

        # 4. RoPE
        cos_half = freqs_cis.real.to(query_states.dtype); sin_half = freqs_cis.imag.to(query_states.dtype)
        cos = torch.cat((cos_half, cos_half), dim=-1); sin = torch.cat((sin_half, sin_half), dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0); sin = sin.unsqueeze(0).unsqueeze(0)
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        # print_stats(query_states, "4. Q after RoPE")
        # print_stats(key_states, "4. K after RoPE")

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
        # print_stats(key_states, "5. K after GQA repeat_kv")

        # 6. 注意力得分
        scaling = head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        # print_stats(attn_weights, "6. Attn Scores (before mask)")

        # # 7. 遮罩
        mask_fill_value = torch.finfo(torch.float16).min
    
        if s > 1:
            causal_mask = torch.triu(torch.full((s, s), mask_fill_value, device=attn_weights.device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + causal_mask
        if attention_mask is not None:
            padding_mask = (attention_mask.data == 0).view(b, 1, 1, s)
            attn_weights = attn_weights.masked_fill(padding_mask, mask_fill_value)
        mask_fill_value = torch.finfo(torch.float16).min
        padding_mask = ~(attention_mask.data.bool()).view(b, 1, 1, s)
        attn_weights = attn_weights.masked_fill(padding_mask, mask_fill_value)
        if attention_mask is not None:
            padding_mask = ~(attention_mask.data.bool()).view(b, 1, 1, s)
            attn_weights = attn_weights.masked_fill(padding_mask, mask_fill_value)
        # print_stats(attn_weights, "7. Attn Scores (after mask)")
        # if s > 1:
        #     causal_mask = torch.full((s, s), mask_fill_value, device=attn_weights.device)
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        #     attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # row_all_inf = (attn_weights.max(dim=-1).values == mask_fill_value).any()
        # if row_all_inf:
        #     print("WARNING: Some rows fully masked!")

        # 8. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # print_stats(attn_weights, "8. Attn Weights (after softmax)")

        # 9. 应用于V
        attn_output = torch.matmul(attn_weights, value_states)
        # print_stats(attn_output, "9. Attn Output (after V matmul)")

        # 10. 输出 reshape 和投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(b, s, h)
        value = F.linear(attn_output, w_out.data) + residual
        # print_stats(value, "10. Final Output (after O-Proj & Residual)")
        # exit()

        # ... (后面的cache逻辑保持不变) ...
        if donate[0]: inputs.delete(); 
        if donate[1]: attention_mask.delete()
        k = k_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)
        v = v_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_kv_heads, head_dim)
        # print(f"k: {k.shape}, v: {v.shape}")
        # exit()
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

        residual = inputs.data
        hidden = self.rms_norm(inputs.data, w_ln.data)

        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)

        query_states = q.view(b, tgt_s, n_head, head_dim).transpose(1, 2)
        key_states = k.view(b, tgt_s, n_kv_heads, head_dim).transpose(1, 2)
        value_states = v.view(b, tgt_s, n_kv_heads, head_dim).transpose(1, 2)

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
        
        # Correctly prepare k_new and v_new for the cache
        k_new = key_states.transpose(1, 2).contiguous().view(b, tgt_s, n_kv_heads, head_dim).permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_heads, head_dim)
        v_new = value_states.transpose(1, 2).contiguous().view(b, tgt_s, n_kv_heads, head_dim).permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_heads, head_dim)

        if isinstance(k_cache, TorchTensor):
            if compress_cache:
                k_cached = k_cache.device.decompress(k_cache)[:src_s-tgt_s]
                v_cached = v_cache.device.decompress(v_cache)[:src_s-tgt_s]
            else:
                k_cached = k_cache.data[:src_s-tgt_s]
                v_cached = v_cache.data[:src_s-tgt_s]
            
            k_all = torch.cat([k_cached, k_new], dim=0)
            v_all = torch.cat([v_cached, v_new], dim=0)
        else:
            k_all = k_new
            v_all = v_new
        
        # 直接从 k_all 张量获取其真实的序列长度
        actual_seq_len = k_all.shape[0]
        # 使用真实长度进行 view 操作，并为保证内存连续性添加 .contiguous()
        key_states = k_all.permute(1, 0, 2).contiguous().view(b, n_kv_heads, actual_seq_len, head_dim)
        value_states = v_all.permute(1, 0, 2).contiguous().view(b, n_kv_heads, actual_seq_len, head_dim)

        # key_states = k_all.permute(1, 0, 2).view(b, n_kv_heads, src_s, head_dim)
        # value_states = v_all.permute(1, 0, 2).view(b, n_kv_heads, src_s, head_dim)

        def repeat_kv_official(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            batch, num_key_value_heads, slen, head_dim_inner = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim_inner)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim_inner)

        key_states = repeat_kv_official(key_states, n_head // n_kv_heads)
        value_states = repeat_kv_official(value_states, n_head // n_kv_heads)

        scaling = head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        
        if attention_mask is not None:
            # padding_mask = (attention_mask.data == 0).view(b, 1, 1, src_s)
            padding_mask = (attention_mask.data[:, :actual_seq_len] == 0).view(b, 1, 1, actual_seq_len)
            attn_weights = attn_weights.masked_fill(padding_mask, torch.finfo(torch.float16).min)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(b, tgt_s, h)
        value = F.linear(attn_output, w_out.data) + residual

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            k_new = compute_device.compressed_device.compress(k_new, comp_config)
            v_new = compute_device.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, compute_device)
            v_new = TorchTensor.create_from_torch(v_new, compute_device)

        return TorchTensor.create_from_torch(value, compute_device), k_new, v_new
        
    # # 在 LLaMAModelComputation 类中

    # def mha_gen(self, compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
    #         n_head, n_kv_heads, k_cache, v_cache, freqs_cis, donate,
    #         attn_sparsity, compress_cache, comp_config):
    #     """
    #     Multi-head attention (decoding phase) - 此版本修正了GQA相关的维度错误。
    #     """
    #     if w_q.device.device_type == DeviceType.COMPRESSED:
    #         w_q, w_k, w_v, w_out = [w.device.decompress(w) for w in [w_q, w_k, w_v, w_out]]

    #     b, tgt_s, h = inputs.shape
    #     src_s = attention_mask.shape[1]
    #     head_dim = h // n_head

    #     residual = inputs.data
    #     hidden = self.rms_norm(inputs.data, w_ln.data)

    #     q = F.linear(hidden, w_q.data)
    #     k = F.linear(hidden, w_k.data)
    #     v = F.linear(hidden, w_v.data)

    #     query_states = q.view(b, tgt_s, n_head, head_dim).transpose(1, 2)
    #     key_states = k.view(b, tgt_s, n_kv_heads, head_dim).transpose(1, 2)
    #     value_states = v.view(b, tgt_s, n_kv_heads, head_dim).transpose(1, 2)

    #     cos_half = freqs_cis.real.to(query_states.dtype)
    #     sin_half = freqs_cis.imag.to(query_states.dtype)
    #     cos = torch.cat((cos_half, cos_half), dim=-1)
    #     sin = torch.cat((sin_half, sin_half), dim=-1)
    #     cos = cos.unsqueeze(0).unsqueeze(0)
    #     sin = sin.unsqueeze(0).unsqueeze(0)
        
    #     def rotate_half(x):
    #         x1 = x[..., : x.shape[-1] // 2]
    #         x2 = x[..., x.shape[-1] // 2 :]
    #         return torch.cat((-x2, x1), dim=-1)

    #     query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    #     key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        
    #     k_new = key_states.transpose(1,2).contiguous().permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_heads, head_dim)
    #     v_new = value_states.transpose(1,2).contiguous().permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_heads, head_dim)
        
    #     if isinstance(k_cache, TorchTensor):
    #         if compress_cache:
    #             k_cached = k_cache.device.decompress(k_cache)[:src_s-tgt_s]
    #             v_cached = v_cache.device.decompress(v_cache)[:src_s-tgt_s]
    #         else:
    #             k_cached = k_cache.data[:src_s-tgt_s]
    #             v_cached = v_cache.data[:src_s-tgt_s]
            
    #         k_all = torch.cat([k_cached, k_new], dim=0)
    #         v_all = torch.cat([v_cached, v_new], dim=0)
    #     else:
    #         k_all = k_new
    #         v_all = v_new
        
    #     # --- !!! 以下是针对本次报错的最终修正 !!! ---
    #     # 修正点：从 cache 恢复时，删除多余的 .transpose(1, 2)，确保维度顺序为 (b, n_kv_heads, s, d)
    #     key_states = k_all.permute(1, 0, 2).view(b, n_kv_heads, src_s, head_dim)
    #     value_states = v_all.permute(1, 0, 2).view(b, n_kv_heads, src_s, head_dim)
    #     # ------------------- 修正结束 -------------------

    #     def repeat_kv_official(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    #         batch, num_key_value_heads, slen, head_dim_inner = hidden_states.shape
    #         if n_rep == 1:
    #             return hidden_states
    #         hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim_inner)
    #         return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim_inner)

    #     # 现在 key_states (1, 8, S, 128) -> repeat_kv -> (1, 32, S, 128)
    #     key_states = repeat_kv_official(key_states, n_head // n_kv_heads)
    #     value_states = repeat_kv_official(value_states, n_head // n_kv_heads)

    #     scaling = head_dim ** -0.5
    #     # 现在 query (1, 32, 1, 128) 和 key (1, 32, S, 128) 的头数量匹配，可以计算了
    #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        
    #     # if attention_mask is not None:
    #     #     padding_mask = (attention_mask.data == 0).view(b, 1, 1, src_s)
    #     #     attn_weights = attn_weights.masked_fill(padding_mask, torch.finfo(attn_weights.dtype).min)
    #     # 7. 应用注意力遮罩
    #     if attention_mask is not None:
    #         padding_mask = (attention_mask.data == 0).view(b, 1, 1, src_s)
    #         # 强制使用 float16 的最小值进行填充
    #         attn_weights = attn_weights.masked_fill(padding_mask, torch.finfo(torch.float16).min)

    #     attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #     attn_output = torch.matmul(attn_weights, value_states)

    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.reshape(b, tgt_s, h)
    #     value = F.linear(attn_output, w_out.data) + residual

    #     if donate[0]: inputs.delete()
    #     if donate[1]: attention_mask.delete()

    #     if compress_cache:
    #         k_new = compute_device.compressed_device.compress(k_new, comp_config)
    #         v_new = compute_device.compressed_device.compress(v_new, comp_config)
    #     else:
    #         k_new = TorchTensor.create_from_torch(k_new, compute_device)
    #         v_new = TorchTensor.create_from_torch(v_new, compute_device)

    #     return TorchTensor.create_from_torch(value, compute_device), k_new, v_new
    
    
    # def mha_gen_old(self, compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
    #             n_head, n_kv_heads, k_cache, v_cache, freqs_cis, donate,
    #             attn_sparsity, compress_cache, comp_config):
    #     """Multi-head attention (decoding phase) with RoPE and GQA"""
    #     # decompress weights
    #     if w_q.device.device_type == DeviceType.COMPRESSED:
    #         w_q = w_q.device.decompress(w_q)
    #         w_k = w_k.device.decompress(w_k)
    #         w_v = w_v.device.decompress(w_v)
    #         w_out = w_out.device.decompress(w_out)

    #     b, tgt_s, h = inputs.shape
    #     src_s = attention_mask.shape[1]
    #     head_dim = h // n_head
    #     scaling = head_dim ** -0.5

    #     # Pre-normalization with RMSNorm
    #     residual = inputs.data
    #     hidden = self.rms_norm(inputs.data, w_ln.data)

    #     # Linear projections
    #     q = F.linear(hidden, w_q.data) * scaling
    #     k = F.linear(hidden, w_k.data)
    #     v = F.linear(hidden, w_v.data)
        
    #     # Reshape
    #     q = q.view(b, tgt_s, n_head, head_dim)
    #     k = k.view(b, tgt_s, n_kv_heads, head_dim)
    #     v = v.view(b, tgt_s, n_kv_heads, head_dim)

    #     # Apply RoPE
    #     q, k = self.apply_rotary_emb(q, k, freqs_cis)

    #     # Save new k,v for cache
    #     k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_heads, head_dim)
    #     v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_heads, head_dim)

    #     # Load cached k,v and concatenate with new
    #     if isinstance(k_cache, TorchTensor):
    #         if compress_cache:
    #             k_cached = k_cache.device.decompress(k_cache)[:src_s-1]
    #             v_cached = v_cache.device.decompress(v_cache)[:src_s-1]
    #         else:
    #             k_cached = k_cache.data[:src_s-1]
    #             v_cached = v_cache.data[:src_s-1]
            
    #         k_all = torch.cat([k_cached, k_new], dim=0)  # (src_s, b*n_kv_heads, head_dim)
    #         v_all = torch.cat([v_cached, v_new], dim=0)
    #     else:
    #         # Handle mixed device case if needed
    #         k_all = k_new
    #         v_all = v_new

    #     # Reshape for attention computation
    #     k_all = k_all.permute(1, 2, 0).reshape(b * n_kv_heads, head_dim, src_s)
    #     v_all = v_all.permute(1, 0, 2).reshape(b * n_kv_heads, src_s, head_dim)

    #     # Repeat k,v for grouped query attention
    #     num_key_value_groups = n_head // n_kv_heads
    #     if num_key_value_groups > 1:
    #         k_all = k_all.view(b, n_kv_heads, head_dim, src_s).repeat_interleave(num_key_value_groups, dim=1).view(b * n_head, head_dim, src_s)
    #         v_all = v_all.view(b, n_kv_heads, src_s, head_dim).repeat_interleave(num_key_value_groups, dim=1).view(b * n_head, src_s, head_dim)
    #     else:
    #         k_all = k_all.view(b * n_head, head_dim, src_s)
    #         v_all = v_all.view(b * n_head, src_s, head_dim)

    #     # Compute attention
    #     q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        
    #     if k_all.is_cuda:
    #         scores = torch.matmul(q, k_all)
    #         mask = attention_mask.data.view(b, 1, 1, src_s).expand(-1, n_head, -1, -1).reshape(b * n_head, 1, src_s)
    #         scores = torch.where(mask, scores, -1e4)
    #         attn_weights = F.softmax(scores, dim=2)
    #         value = torch.matmul(attn_weights, v_all).view(b, n_head, tgt_s, head_dim)
    #     else:
    #         # CPU computation
    #         q = q.float().cpu()
    #         k_all, v_all = k_all.float(), v_all.float()
    #         scores = torch.matmul(q, k_all)
    #         mask = attention_mask.data.view(b, 1, 1, src_s).expand(-1, n_head, -1, -1).reshape(b * n_head, 1, src_s)
    #         scores = torch.where(mask, scores, -1e4)
    #         attn_weights = F.softmax(scores, dim=2)
    #         value = torch.matmul(attn_weights, v_all).view(b, n_head, tgt_s, head_dim).cuda().half()

    #     # Output projection
    #     value = value.transpose(1, 2).view(b, tgt_s, h)
    #     value = F.linear(value, w_out.data) + residual

    #     if donate[0]: inputs.delete()
    #     if donate[1]: attention_mask.delete()

    #     if compress_cache:
    #         k_new = compute_device.compressed_device.compress(k_new, comp_config)
    #         v_new = compute_device.compressed_device.compress(v_new, comp_config)
    #     else:
    #         k_new = TorchTensor.create_from_torch(k_new, compute_device)
    #         v_new = TorchTensor.create_from_torch(v_new, compute_device)

    #     return TorchTensor.create_from_torch(value, compute_device), k_new, v_new

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
        print_stats(out, "11a. After Post-Attention RMSNorm")
        
        # SwiGLU: gate(x) * SiLU(up(x))
        # 检查权重是否正确加载
        if hasattr(self, 'debug_data'):
            self.debug_data['mlp_weights'] = {
                'gate_weight_shape': w_gate.data.shape,
                'gate_weight_range': [w_gate.data.min().item(), w_gate.data.max().item()],
                'gate_weight_mean': w_gate.data.mean().item(),
                'up_weight_shape': w_up.data.shape,
                'up_weight_range': [w_up.data.min().item(), w_up.data.max().item()],
                'up_weight_mean': w_up.data.mean().item(),
                'down_weight_shape': w_down.data.shape,
                'down_weight_range': [w_down.data.min().item(), w_down.data.max().item()],
                'down_weight_mean': w_down.data.mean().item(),
            }
        
        gate = F.linear(out, w_gate.data)
        print_stats(gate, "11a1. Gate projection output")
        
        up = F.linear(out, w_up.data)
        print_stats(up, "11a2. Up projection output")
        
        silu_gate = F.silu(gate)  # SiLU activation
        print_stats(silu_gate, "11a3. After SiLU activation")
        
        intermediate = silu_gate * up  # Element-wise multiplication
        print_stats(intermediate, "11a4. SiLu(Gate) * up intermediate")
        
        # Down projection
        out = F.linear(intermediate, w_down.data)
        print_stats(out, "11a5. After down projection (before residual)")
        
        out = out + residual  # Residual connection
        print_stats(out, "11b. After MLP (with residual)")

        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, compute_device)


class LLaMAInputEmbed(BaseModelLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LLaMAModelComputation'):
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
        if i == 0: # 输入嵌入只在prefill阶段运行和记录
            # 打印输入嵌入的统计信息，格式与test_final_verify_and_trace.py一致
            print_stats(h.data, "0. Initial Input Embeddings")
            
            FLEXGEN_DEBUG_LOGGER.record(
                tensor_data=h.data,
                gen_token_step=i,
                layer_id=0,
                layer_name="input_embed"
            )


class LLaMAOutputEmbed(BaseModelLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LLaMAModelComputation'):
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
                 computation: 'LLaMAModelComputation', 
                 layer_id: int):
        super().__init__(config, env, policy, weight_map=weight_map) 
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)  
        self.computation = computation 
        self.layer_id = layer_id

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
            # # Query projection
            # ((h, h), "attn_q_proj_weight", p / self.weight_map["q_proj"][0], self.config.dtype),
            # # Key projection (potentially smaller for GQA)
            # ((kv_dim, h), "attn_k_proj_weight", p / self.weight_map["k_proj"][0], self.config.dtype),
            # # Value projection (potentially smaller for GQA)
            # ((kv_dim, h), "attn_v_proj_weight", p / self.weight_map["v_proj"][0], self.config.dtype),
            # # Output projection
            # ((h, h), "attn_o_proj_weight", p / self.weight_map["o_proj"][0], self.config.dtype),
            # # RMSNorm weight (no bias)
            # ((h,), "attn_norm_weight", p / self.weight_map["norm"][0], self.config.dtype),
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
            correct_cache_len = self.task.prompt_len + i - 1
            if k_cache.data.shape[0] != correct_cache_len:
                # 手动将从底层框架得到的错误尺寸张量，裁剪为正确的尺寸
                k_cache.data = k_cache.data[:correct_cache_len]
                v_cache.data = v_cache.data[:correct_cache_len]
            # if i == 1: # 只在生成第一个新token时记录，避免日志过多
            #     print("-" * 25, f" 调试信息: Layer {self.layer_id}, Step {i} ", "-" * 25)
            #     print(f"i: {i}, k_cache: {k_cache.shape}, v_cache: {v_cache.shape}, prompt_len: {self.task.prompt_len}")
                
            #     # 检查 K-Cache
            #     k_data = k_cache.data
            #     # 沿着 head 和 head_dim 维度计算每行的绝对值之和
            #     # 如果一整行都是0，那么它的和也为0
            #     k_row_sums = torch.abs(k_data).sum(dim=(1, 2))
            #     # 计算和不为0的行的数量
            #     k_nonzero_rows = torch.count_nonzero(k_row_sums).item()
                
            #     print(f"K-Cache 总形状: {k_data.shape}")
            #     print(f"检测到 {k_nonzero_rows} / {k_data.shape[0]} 行包含非零数据。")

            #     # 检查 V-Cache (同理)
            #     v_data = v_cache.data
            #     v_row_sums = torch.abs(v_data).sum(dim=(1, 2))
            #     v_nonzero_rows = torch.count_nonzero(v_row_sums).item()

            #     print(f"V-Cache 总形状: {v_data.shape}")
            #     print(f"检测到 {v_nonzero_rows} / {v_data.shape[0]} 行包含非零数据。")
            #     print("-" * 80)

            #     FLEXGEN_DEBUG_LOGGER.record(
            #         tensor_data=k_cache.data,
            #         gen_token_step=i,
            #         layer_id=self.layer_id + 1, # 使用您在 LLaMATransformerLayer 中添加的 layer_id
            #         layer_name="mha_gen_input_k_cache"
            #     )
            #     FLEXGEN_DEBUG_LOGGER.record(
            #         tensor_data=v_cache.data,
            #         gen_token_step=i,
            #         layer_id=self.layer_id + 1,
            #         layer_name="mha_gen_input_v_cache"
            #     )
            h, new_k_cache, new_v_cache = self.computation.mha_gen(
                self.compute_device, h, mask, w_q, w_k, w_v, w_out, w_ln,
                self.n_head, self.n_kv_heads, k_cache, v_cache, freqs_cis, donate,
                self.policy.attn_sparsity, self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h
        # save_debug_data(h, i, 0, "mha")


class LLaMAMLP(BaseModelLayer):
    def __init__(self, 
                 config: FlexModelConfig, 
                 env: ExecutionEnv, 
                 policy: Policy, 
                 weight_map: dict, 
                 computation: 'LLaMAModelComputation'):
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
            # ((intermediate_size, h), "mlp_gate_proj_weight", p / self.weight_map["gate_proj"][0], self.config.dtype),
            # ((intermediate_size, h), "mlp_up_proj_weight", p / self.weight_map["up_proj"][0], self.config.dtype),
            # ((h, intermediate_size), "mlp_down_proj_weight", p / self.weight_map["down_proj"][0], self.config.dtype),
            # ((h,), "mlp_norm_weight", p / self.weight_map["norm"][0], self.config.dtype),
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
                 computation: 'LLaMAModelComputation', 
                 layer_id: int):
        super().__init__(config, env, policy, weight_map)
        self.layer_id = layer_id # <--- 保存 layer_id
        self.attention = LLaMASelfAttention(config=config, env=env, policy=policy, 
                                           weight_map=weight_map['attention'], computation=computation, layer_id=layer_id)
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

        # 只在prefill阶段(i==0)且是第一层或第二层时打印详细信息
        if i == 0 and self.layer_id in [0, 1]:
            print(f"\n\n{'='*80}\n--- TRACING LAYER {self.layer_id} --- \n{'='*80}\n")
            
            # 记录输入状态
            print_stats(hidden.val.data, f"Layer {self.layer_id} - 0. Initial Hidden State for Layer {self.layer_id}")

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k)
        if i == 0: # 只在prefill阶段检查
            FLEXGEN_DEBUG_LOGGER.record(
                tensor_data=hidden.val.data,
                gen_token_step=i,
                layer_id=self.layer_id + 1,
                layer_name="attention_output_only" # 使用一个独特的层名
            )
            
            # 只在第一层或第二层时打印attention输出
            if self.layer_id in [0, 1]:
                print_stats(hidden.val.data, f"Layer {self.layer_id} - 10. Output after Attention Block")
        
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)

        FLEXGEN_DEBUG_LOGGER.record(
            tensor_data=hidden.val.data, # .cpu() 会在record方法内部完成
            gen_token_step=i,
            layer_id=self.layer_id + 1,
            layer_name="transformer_output"
        )
        
        # 只在第一层或第二层时打印最终输出
        if i == 0 and self.layer_id in [0, 1]:
            print_stats(hidden.val.data, f"Layer {self.layer_id} - 11c. Final Output of Layer {self.layer_id}")
        # if self.layer_id == 1:
        #     exit()


class LLaMAModel(BaseModel):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 path:str):
        super().__init__(config, env, policy, weight_map=weight_map, path=path) 

        self.computation = LLaMAModelComputation()
        
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
                self.layers.append(LLaMATransformerLayer(self.config, self.env, self.policy, self.weight_map['layers'][layer_id], self.computation, layer_id=layer_id))
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


    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        # print(task.__dict__)
        tmp_batch_size = None
        if self.policy.gpu_batch_size * self.num_gpu_batches != len(task.inputs):
            tmp_batch_size = self.policy.gpu_batch_size
            self.policy.gpu_batch_size = len(task.inputs)
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "bos_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0  # 最终fallback
            
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()
        if tmp_batch_size is not None:
            self.policy.gpu_batch_size = tmp_batch_size

        # !!! 新增代码：在函数末尾统一保存所有记录的数据 !!!
        FLEXGEN_DEBUG_LOGGER.save_to_disk()
        print("All results have been saved to disk.")
        return self.output_ids
    
    
    # def save_debug_data(self, filename_prefix="flexgen_debug"):
    #     """保存调试数据到文件"""
    #     import os
    #     import json
    #     import pickle
        
    #     # 创建debug_results目录
    #     os.makedirs('debug_results', exist_ok=True)
        
    #     # 保存统计信息到JSON
    #     stats_only = {}
    #     if 'input_hidden_state' in self.debug_data:
    #         stats_only['input_hidden_state'] = {
    #             'shape': self.debug_data['input_hidden_state']['shape'],
    #             'range': self.debug_data['input_hidden_state']['range'],
    #             'mean': self.debug_data['input_hidden_state']['mean'],
    #             'std': self.debug_data['input_hidden_state']['std']
    #         }
        
    #     if 'normalized_hidden_state' in self.debug_data:
    #         stats_only['normalized_hidden_state'] = {
    #             'range': self.debug_data['normalized_hidden_state']['range'],
    #             'mean': self.debug_data['normalized_hidden_state']['mean'],
    #             'std': self.debug_data['normalized_hidden_state']['std']
    #         }
        
    #     if 'logits' in self.debug_data:
    #         stats_only['logits'] = {
    #             'shape': self.debug_data['logits']['shape'],
    #             'range': self.debug_data['logits']['range'],
    #             'mean': self.debug_data['logits']['mean'],
    #             'std': self.debug_data['logits']['std'],
    #             'last_token_logits': {
    #                 'shape': self.debug_data['logits']['last_token_logits']['shape'],
    #                 'range': self.debug_data['logits']['last_token_logits']['range'],
    #                 'mean': self.debug_data['logits']['last_token_logits']['mean'],
    #                 'std': self.debug_data['logits']['last_token_logits']['std']
    #             }
    #         }
        
    #     if 'top_k' in self.debug_data:
    #         stats_only['top_k'] = {
    #             'logits': self.debug_data['top_k']['logits'].tolist(),
    #             'indices': self.debug_data['top_k']['indices'].tolist()
    #         }
        
    #     # 保存统计信息
    #     stats_filename = f'debug_results/{filename_prefix}_stats.json'
    #     with open(stats_filename, 'w') as f:
    #         json.dump(stats_only, f, indent=2)
        
    #     # 保存完整张量数据
    #     tensors_filename = f'debug_results/{filename_prefix}_tensors.pkl'
    #     with open(tensors_filename, 'wb') as f:
    #         pickle.dump(self.debug_data, f)
        
    #     print(f"FlexGen调试数据已保存:")
    #     print(f"  统计信息: {stats_filename}")
    #     print(f"  完整张量: {tensors_filename}")
    #     print(f"  JSON文件大小: {os.path.getsize(stats_filename) / 1024:.2f} KB")
    #     print(f"  Pickle文件大小: {os.path.getsize(tensors_filename) / (1024*1024):.2f} MB")
        
    #     return stats_filename, tensors_filename
