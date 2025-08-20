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
    def _attention_weights(self, compute_device, q, k, mask):
        #  """
            # Compute attention weights using torch.matmul, no reshape needed.
            # Args:
            #     q: (b, n_head, 1, head_dim)
            #     k: (b, n_head, src_s, head_dim)
            #     mask: (b, 1, 1, src_s) or (b, 1, src_s) -> will be broadcasted
            # Returns:
            #     attn_weights: (b, n_head, 1, src_s)
            # """
        # QK^T: (b, n_head, 1, head_dim) @ (b, n_head, src_s, head_dim)^T
        # We need to transpose k: -> (b, n_head, head_dim, src_s)
        k_t = k.transpose(-2, -1)  # (b, n_head, head_dim, src_s)
        
        # Matmul: (b, n_head, 1, head_dim) @ (b, n_head, head_dim, src_s) -> (b, n_head, 1, src_s)
        attn_weights = torch.matmul(q, k_t)  # (b, n_head, 1, src_s)

        # Scale by sqrt(d_k)
        head_dim = q.size(-1)
        attn_weights = attn_weights / (head_dim ** 0.5)

        # Expand mask if needed
        if mask.dim() == 3:  # (b, 1, src_s)
            mask = mask.unsqueeze(1)  # (b, 1, 1, src_s)

        # Apply mask
        attn_weights = torch.where(mask, attn_weights, torch.tensor(-1e4, dtype=attn_weights.dtype, device=attn_weights.device))

        # Softmax over last dimension (src_s)
        attn_weights = F.softmax(attn_weights, dim=-1)

        return attn_weights  # (b, n_head, 1, src_s)

    def _attention_value(self, compute_device, attn_weights, v):
        # """
        # Compute attention output: attn_weights @ V
        # Args:
        #     attn_weights: (b, n_head, 1, src_s)
        #     v: (b, n_head, src_s, head_dim)
        # Returns:
        #     output: (b, n_head, 1, head_dim)
        # """

        # attn_weights @ V: (b, n_head, 1, src_s) @ (b, n_head, src_s, head_dim) -> (b, n_head, 1, head_dim)
        attn_output = torch.matmul(attn_weights, v)  # (b, n_head, 1, head_dim)

        return attn_output
    
    def _sparse_attention_value(self, compute_device, q, k_cache, v_new, v_cache, 
                           mask, bsz, src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # """
        # Sparse attention using general_copy, keep (b, n_head, s, head_dim) structure.
        
        # Args:
        #     q: (b, n_head, 1, head_dim)
        #     k_cache: TorchTensor (b, n_head, src_s, head_dim) - cached keys
        #     v_new: (b, n_head, 1, head_dim)
        #     v_cache: TorchTensor (b, n_head, src_s, head_dim) - cached values
        #     mask: (b, 1, src_s)
        #     attn_sparsity: float
        
        # Returns:
        #     output: (b, n_head, 1, head_dim)
        # """
        device = q.device

        # Step 1: 计算完整注意力权重
        # k_cache: (b, n_head, src_s, head_dim)
        # 但注意：当前 token 的 k 还没更新，所以先用缓存的
        attn_weights_full = self._attention_weights(q, k_cache, mask)  # (b, n_head, 1, src_s)

        # Step 2: 分离历史和当前
        # attn_weights: (b, n_head, 1, src_s)
        # 历史部分: 前 src_s-1 个
        attn_weights_hist = attn_weights_full[..., :-1]  # (b, n_head, 1, src_s - 1)
        attn_weight_curr = attn_weights_full[..., -1:]   # (b, n_head, 1, 1)

        # 计算 topk
        topk = max(1, int(attn_sparsity * (src_s - 1)))

        # 获取 topk indices: (b, n_head, 1, topk)
        topk_weights, topk_indices = torch.topk(attn_weights_hist, k=topk, dim=-1, sorted=False)
        # topk_indices: (b, n_head, 1, topk) -> reshape for general_copy: (topk, b * n_head)
        topk_indices_flat = topk_indices.squeeze(2).permute(2, 0, 1)  # (topk, b, n_head)
        topk_indices_flat = topk_indices_flat.reshape(topk, bsz * n_head).transpose(0, 1)  # (b * n_head, topk)

        # Step 3: 使用 general_copy 从 v_cache 中拷贝 topk values
        # v_cache: (b, n_head, src_s, head_dim) -> reshape to (src_s, b * n_head, head_dim)
        v_cache_3d = v_cache.permute(2, 0, 1, 3).reshape(src_s, bsz * n_head, head_dim)

        # 分配 buffer: (topk, b * n_head, head_dim)
        v_buf = compute_device.allocate((topk, bsz * n_head, head_dim), dtype=np.float16)
        topk_indices_cpu = topk_indices_flat.cpu()

        # 拷贝: v_buf[i, :, :] = v_cache_3d[topk_indices_cpu[i], range(b*n_head), :]
        indices_src = topk_indices_cpu  # (b * n_head, topk)
        indices_tgt = (slice(0, topk), slice(0, bsz * n_head))
        general_copy(v_buf, indices_tgt, v_cache_3d, indices_src)
        compute_device.synchronize()

        # Step 4: 构造稀疏 v: [topk historical v] + [current v_new]
        # v_buf: (topk, b * n_head, head_dim) -> (b * n_head, topk, head_dim)
        v_sparse = v_buf.data.transpose(0, 1).reshape(bsz, n_head, topk, head_dim)  # (b, n_head, topk, head_dim)

        # 拼接当前 v_new: (b, n_head, 1, head_dim)
        v_sparse = torch.cat([v_sparse, v_new], dim=-2)  # (b, n_head, topk+1, head_dim)

        # Step 5: 构造稀疏 attention weights
        attn_weights_sparse = torch.cat([topk_weights, attn_weight_curr], dim=-1)  # (b, n_head, 1, topk+1)

        # Step 6: 计算输出
        output = torch.matmul(attn_weights_sparse, v_sparse)  # (b, n_head, 1, head_dim)

        return output

    def _mixed_device_attention(self, compute_device, q, k_cache, v_cache, 
                        k_new, v_new, mask, bsz, src_s, tgt_s, n_head, head_dim):
        # """
        # Mixed device attention: GPU and CPU parts.
        
        # k_cache: (k_gpu, k_cpu), each (b_part, n_head, src_s, head_dim)
        # """
        k_gpu, k_cpu = k_cache
        v_gpu, v_cpu = v_cache

        b_gpu = k_gpu.shape[0]
        b_cpu = bsz - b_gpu

        q_gpu = q[:b_gpu]
        q_cpu = q[b_gpu:].float().cpu()

        k_new_gpu = k_new[:b_gpu]
        k_new_cpu = k_new[b_gpu:].float().cpu()

        v_new_gpu = v_new[:b_gpu]
        v_new_cpu = v_new[b_gpu:].float().cpu()

        mask_gpu = mask[:b_gpu]
        mask_cpu = mask[b_gpu:]

        # --- GPU Part ---
        k_gpu_updated = k_gpu.clone()
        k_gpu_updated[:, :, src_s-1:src_s] = k_new_gpu
        v_gpu_updated = v_gpu.clone()
        v_gpu_updated[:, :, src_s-1:src_s] = v_new_gpu

        attn_weights_gpu = self._attention_weights(q_gpu, k_gpu_updated, mask_gpu)
        value_gpu = self._attention_value(attn_weights_gpu, v_gpu_updated)

        # --- CPU Part ---
        k_cpu_updated = k_cpu.clone()
        k_cpu_updated[:, :, src_s-1:src_s] = k_new_cpu
        v_cpu_updated = v_cpu.clone()
        v_cpu_updated[:, :, src_s-1:src_s] = v_new_cpu

        attn_weights_cpu = self._attention_weights(q_cpu, k_cpu_updated, mask_cpu)
        value_cpu = self._attention_value(attn_weights_cpu, v_cpu_updated)

        # Move back
        value_cpu = value_cpu.cuda().half()

        # Concat on batch dim
        value = torch.cat([value_gpu, value_cpu], dim=0)  # (bsz, n_head, 1, head_dim)

        return value
    
    def input_embed(self, compute_device: TorchDevice, inputs: TorchTensor, w_token: TorchTensor, pad_token_id: int, donate: list):
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        if donate[0]: inputs.delete()
        
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        return TorchTensor.create_from_torch(token_embed, compute_device)
    
    
    # SelfAttention Prefill
    def mha(self, compute_device: TorchDevice, 
            hidden_states: TorchTensor, attention_mask: TorchTensor,
            w_q: TorchTensor, w_k: TorchTensor, w_v: TorchTensor, w_o: TorchTensor, w_norm: TorchTensor,
            n_head: int, config: FlexModelConfig, donate: list, compress_cache: bool, comp_config: any, ):
        
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q, w_k, w_v, w_o, w_norm = [x.device.decompress(x) for x in [w_q, w_k, w_v, w_o, w_norm]]

      
        bsz, seq_len, h = hidden_states.shape
        n_q_head = n_head
        n_kv_head = getattr(config, 'num_key_value_heads', n_q_head)
        num_key_value_groups = n_q_head // n_kv_head
        head_dim = h // n_q_head
        
        residual = hidden_states.data
        hidden_norm = self.rms_norm(residual, w_norm.data, config.rms_norm_eps)
        
        # shape: (b, s, h)
        q = F.linear(hidden_norm, w_q.data)
        k = F.linear(hidden_norm, w_k.data)
        v = F.linear(hidden_norm, w_v.data)

        # 重塑为多头形式
        # shape: (b, s, n_head, head_dim)
        q = q.view(bsz, seq_len, n_q_head, head_dim)
        k = k.view(bsz, seq_len, n_kv_head, head_dim)
        v = v.view(bsz, seq_len, n_kv_head, head_dim)

        # 应用RoPE
        q, k = self.apply_rotary_emb(q, k, freqs_cis=compute_device.rotary_emb_cis[:seq_len])

        # 先将原始的 K 和 V 保存起来, 以便之后存入缓存
        k_for_cache = k 
        v_for_cache = v
        
        # 重复kv头以匹配q头数
        # shape:(b, s, n_head, head_dim)
        # if num_key_value_groups > 1:
        #   n_head = n_head*num_key_value_groups
        # else:
        #     n_head = n_head
        k = self.repeat_kv(k, num_key_value_groups)
        v = self.repeat_kv(v, num_key_value_groups)

        # 转置后注意力计算
        # shape: (b, n_head, s*, head_dim)
        # q: s* is seq_len
        # keys, values: s* is cache_len + seq_len
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # shape: (b, n_head, s_q, s_keys)
        scores = torch.matmul(q, k.transpose(2, 3)) / (head_dim ** 0.5)

        # 计算注意力分数
        # mask shape is (b, s_q)
        mask = attention_mask.data
        if mask is not None:
            scores = scores + mask.unsqueeze(1).unsqueeze(-1)
        
        # 应用因果掩码 (causal mask) 防止看到未来 token
        # 只有在 prefill 阶段 (seq_len > 1) 才需要
        if seq_len > 1:
            # 创建一个上三角矩阵, 对角线以上的位置被设置为-inf
            causal_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=scores.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            scores = scores + causal_mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        # shape: (b, n_head, s, s)
        output = torch.matmul(scores, v)
        # shape: (b, s, n_head, s)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        # return F.linear(output, w_o.data)
        output = F.linear(output, w_o.data) + residual
        

        if donate[0]: hidden_states.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            k_to_cache = compute_device.compressed_device.compress(k_for_cache, comp_config)
            v_to_cache = compute_device.compressed_device.compress(v_for_cache, comp_config)
        else:
            k_to_cache = TorchTensor.create_from_torch(k_for_cache, compute_device)
            v_to_cache = TorchTensor.create_from_torch(v_for_cache, compute_device)

        return TorchTensor.create_from_torch(output, compute_device), k_to_cache, v_to_cache

    # SelfAttention Decode
    def mha_gen(self, compute_device: TorchDevice, hidden_states: TorchTensor, attention_mask: TorchTensor,
                w_q: TorchTensor, w_k: TorchTensor, w_v: TorchTensor, w_o: TorchTensor, w_norm: TorchTensor,
                n_head: int, config: FlexModelConfig, donate: list, compress_cache: bool, comp_config: any,
                attn_sparsity:Union[int, float], k_cache: Union[TorchTensor, ValueHolder], v_cache: Union[TorchTensor, ValueHolder]):

        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q, w_k, w_v, w_o, w_norm = [x.device.decompress(x) for x in [w_q, w_k, w_v, w_o, w_norm]]

        bsz, tgt_s, h = hidden_states.shape
        src_s = attention_mask.shape[1]
        n_q_head = n_head
        n_kv_head = getattr(config, 'num_key_value_heads', n_q_head)
        num_key_value_groups = n_q_head // n_kv_head
        head_dim = h // n_q_head
                       
        residual = hidden_states.data
        hidden_norm = self.rms_norm(residual, w_norm.data, config.rms_norm_eps)

        # 1. 计算当前步的 Q, K, V
        q = F.linear(hidden_norm, w_q.data)
        k_new = F.linear(hidden_norm, w_k.data)
        v_new = F.linear(hidden_norm, w_v.data)

        q = q.view(bsz, tgt_s, n_q_head, head_dim)
        k_new = k_new.view(bsz, tgt_s, n_kv_head, head_dim)
        v_new = v_new.view(bsz, tgt_s, n_kv_head, head_dim)

        q, k_new = self.apply_rotary_emb(q, k_new, freqs_cis=compute_device.rotary_emb_cis[src_s - 1 : src_s])

        # 2. 从 load_cache 加载历史 K, V
        #    k_cache/v_cache 的布局是 (bsz, past_seq_len, n_kv_head, head_dim)
        past_k = k_cache.data
        past_v = v_cache.data
        
        # 3. 将历史 K/V 和 新的 K/V 拼接起来用于本次计算
        #    新的 k_new/v_new 形状是 (bsz, 1, n_kv_head, head_dim)
        keys = torch.cat([past_k, k_new], dim=1)
        values = torch.cat([past_v, v_new], dim=1)

        # 4. 对拼接后的完整 K/V 进行 GQA 扩展和转置
        q_for_attn = q.transpose(1, 2)
        keys_for_attn = self.repeat_kv(keys, num_key_value_groups).transpose(1, 2)
        values_for_attn = self.repeat_kv(values, num_key_value_groups).transpose(1, 2)

        # 5. 执行注意力计算
        if keys_for_attn.is_cuda:
            attn_weights = self._attention_weights(compute_device, q_for_attn, keys_for_attn, attention_mask.data)
            value = self._attention_value(attn_weights, values_for_attn)
        else: # CPU path
            q_cpu = q_for_attn.float().cpu()
            k_cpu, v_cpu = keys_for_attn.float(), values_for_attn.float()
            attn_weights = self._attention_weights(compute_device, q_cpu, k_cpu, attention_mask.data)
            value = self._attention_value(attn_weights, v_cpu).cuda().half()

        # 6. 计算最终输出
        value = value.transpose(1, 2).contiguous().view(bsz, tgt_s, -1)
        value = F.linear(value, w_o.data) + residual
        
        if donate[0]: hidden_states.delete()
        if donate[1]: attention_mask.delete()

        # 7. 返回的 k_new/v_new 是未经任何重复或转置的原始版本，用于写入缓存
        #    它的布局 (bsz, 1, n_kv_head, head_dim) 与缓存期望的布局一致
        if compress_cache:
            k_new_to_cache = compute_device.compressed_device.compress(k_new, comp_config)
            v_new_to_cache = compute_device.compressed_device.compress(v_new, comp_config)
        else:
            k_new_to_cache = TorchTensor.create_from_torch(k_new, compute_device)
            v_new_to_cache = TorchTensor.create_from_torch(v_new, compute_device)

        return TorchTensor.create_from_torch(value, compute_device), k_new_to_cache, v_new_to_cache
    
    
    def mha_gen_old(self, compute_device: TorchDevice, hidden_states: TorchTensor, attention_mask: TorchTensor,
                w_q: TorchTensor, w_k: TorchTensor, w_v: TorchTensor, w_o: TorchTensor, w_norm: TorchTensor,
                n_head: int, config: FlexModelConfig, donate: list, compress_cache: bool, comp_config: any,
                attn_sparsity:Union[int, float], k_cache: Union[TorchTensor, ValueHolder], v_cache: Union[TorchTensor, ValueHolder]):

        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q, w_k, w_v, w_o = [x.device.decompress(x) for x in [w_q, w_k, w_v, w_o]]

        bsz, tgt_s, h = hidden_states.shape
        src_s = attention_mask.shape[1]
        n_q_head = n_head
        n_kv_head = getattr(config, 'num_key_value_heads', n_q_head)
        num_key_value_groups = n_q_head // n_kv_head
        head_dim = h // n_q_head
                       
        residual = hidden_states.data
        hidden_norm = self.rms_norm(residual, w_norm.data, config.rms_norm_eps)

        # shape: (b, tgt_s, h)
        q = F.linear(hidden_norm, w_q.data)
        k_new = F.linear(hidden_norm, w_k.data)
        v_new = F.linear(hidden_norm, w_v.data)

        # shape: (b, tgt_s, n_head, head_dim) n_head*head_dim = h
        q = q.view(bsz, tgt_s, n_q_head, head_dim)
        k_new = k_new.view(bsz, tgt_s, n_kv_head, head_dim)
        v_new = v_new.view(bsz, tgt_s, n_kv_head, head_dim)

        # 应用RoPE
        q, k_new = self.apply_rotary_emb(q, k_new, freqs_cis=compute_device.rotary_emb_cis[src_s - 1 : src_s])

        k_repeated = self.repeat_kv(k_new, num_key_value_groups)
        v_repeated = self.repeat_kv(v_new, num_key_value_groups)

        # # 将原始的 k_new/v_new 保存
        # k_new_for_cache = k_new.transpose(1, 2)
        # v_new_for_cache = v_new.transpose(1, 2)

        # k_repeated = self.repeat_kv(k_new, num_key_value_groups)
        # v_repeated = self.repeat_kv(v_new, num_key_value_groups)

        # 转置后注意力计算
        # shape: (b, n_head, s*, head_dim)
        # q: s* is seq_len
        # keys, values: s* is cache_len + seq_len
        q = q.transpose(1, 2)
        k_repeated = k_repeated.transpose(1, 2)
        v_repeated = v_repeated.transpose(1, 2)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (b, n_head, s, head_dim)
                    k = k_cache.device.decompress(k_cache)[:, :, :src_s]
                    v = v_cache.device.decompress(v_cache)[:, :, :src_s]
                else:
                    # shape: (b, n_head, s, head_dim)
                    k = k_cache.data[:, :, :src_s]
                    v = v_cache.data[:, :, :src_s]
                
                # 更新最新的 kvcache
                # k[:, :, src_s - 1:src_s] = k_new
                # v[:, :, src_s - 1:src_s] = v_new
                k[:, :, src_s - 1:src_s] = k_new.transpose(1,2)
                v[:, :, src_s - 1:src_s] = v_new.transpose(1,2)

                if k.is_cuda:
                    attn_weights = self._attention_weights(compute_device, q, k, attention_mask.data)
                    value = self._attention_value(attn_weights, v)  # (b, n_head, 1, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    attn_weights = self._attention_weights(compute_device, q, k, attention_mask.data)
                    value = self._attention_value(attn_weights, v)  # (b, n_head, 1, head_dim)
            else:  # Sparse attention
                # shape: (b, n_head, s, head_dim)
                k = k_cache.data[:, :, :src_s]
                k[:, :, src_s - 1:src_s] = k_new

                if k.is_cuda:
                    value = self._sparse_attention_value(
                        compute_device, q, k, v_repeated, v_cache,
                        attention_mask.data, bsz, src_s, tgt_s, n_head, head_dim, attn_sparsity
                        )
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(
                        compute_device, q, k, v_repeated, v_cache,
                        attention_mask.data, bsz, src_s, tgt_s, n_head, head_dim, attn_sparsity
                    )
                    value = value.cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(compute_device, q, k_cache, v_cache,
                k_repeated, v_repeated, attention_mask.data, bsz, src_s, tgt_s,
                n_head, head_dim)

        # value shape: (b, n_head, 1, head_dim)
        value = value.transpose(1, 2).contiguous().view(bsz, tgt_s, -1)
        value = F.linear(value, w_o.data) + residual
        
        if donate[0]: hidden_states.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            k_new_for_cache = compute_device.compressed_device.compress(k_new, comp_config)
            v_new_for_cache = compute_device.compressed_device.compress(v_new, comp_config)
        else:
            k_new_for_cache = TorchTensor.create_from_torch(k_new, compute_device)
            v_new_for_cache = TorchTensor.create_from_torch(v_new, compute_device)

        return TorchTensor.create_from_torch(value, compute_device), k_new_for_cache, v_new_for_cache
    
    def mlp(self, compute_device: TorchDevice, hidden_states: TorchTensor,
            w_gate: TorchTensor, w_up: TorchTensor, w_down: TorchTensor, w_norm: TorchTensor,
            config: FlexModelConfig, donate: list):
        
        if w_gate.device.device_type == DeviceType.COMPRESSED:
            w_gate, w_up, w_down, w_norm = [x.device.decompress(x) for x in [w_gate, w_up, w_down, w_norm]]

        residual = hidden_states.data
        hidden_norm = self.rms_norm(hidden_states.data, w_norm.data, config.rms_norm_eps)
        
        # act_fn = ACT2FN[config.hidden_act]
        gate = F.linear(hidden_norm, w_gate.data)
        up = F.linear(hidden_norm, w_up.data)
        down = F.linear(F.silu(gate) * up, w_down.data)
        
        down.add_(residual)
        if donate[0]: hidden_states.delete()
        
        return TorchTensor.create_from_torch(down, compute_device)

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
        
        # s = getattr(self.config, "max_seq_len", None) or getattr(self.config, "max_position_embeddings", None) or 2048
        # shape = (policy.gpu_batch_size, s, self.num_kv_heads, self.head_dim)
        # # print(shape, config.dtype)
        # # exit()
        # self.k_cache = torch.zeros(
        #                             shape, 
        #                             dtype=torch.float16, 
        #                             # device=self.compute_device.dev
        #                               ).cuda()
        # self.v_cache = torch.zeros(
        #                             shape, 
        #                             dtype=torch.float16, 
        #                             # device=self.compute_device.dev
        #                               ).cuda()
        

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
        
        p = Path(converted_path)
        
        weight_specs = [
            ((self.h, self.h), "atten_q_proj", p / self.weight_map["q_proj"][0], self.config.dtype),
            ((self.kv_dim, self.h), "atten_k_proj", p / self.weight_map["k_proj"][0], self.config.dtype),
            ((self.kv_dim, self.h), "atten_v_proj", p / self.weight_map["v_proj"][0], self.config.dtype),
            ((self.h, self.h), "atten_o_proj", p / self.weight_map["o_proj"][0], self.config.dtype),
            ((self.h,), "atten_norm", p / self.weight_map["norm"][0], self.config.dtype),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        w_q, w_k, w_v, w_o, w_norm = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute_device
            weight_read_buf.store((
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_o.smart_copy(dst1),
                w_norm.smart_copy(dst2),
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
        # self.k_cache.zero_()
        # self.v_cache.zero_()
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
        
        current_seq_len = self.task.prompt_len + i -1
        
        # 定义正确的切片: 第0维是batch，第1维是seq
        indices = (slice(0, self.policy.gpu_batch_size), slice(0, current_seq_len))

        if path == 0:  # Direct copy
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
            cpu_indices = (slice(0, self.policy.gpu_batch_size), slice(gpu_k_buf.shape[1], current_seq_len))
            general_copy(k_buf, cpu_indices, k_home, cpu_indices)
            general_copy(v_buf, cpu_indices, v_home, cpu_indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

        # if path == 0:  # Direct copy
        #     # shape: (s, b * n_head, head_dim)
        #     indices = (slice(0, self.task.prompt_len + i),
        #                slice(0, k_home.shape[1]))

        #     if self.policy.attn_sparsity >= 1.0:
        #         cache_read_buf.store((
        #             k_home.smart_copy(dst, indices),
        #             v_home.smart_copy(dst, indices),
        #         ))
        #     else:
        #         cache_read_buf.store((
        #             k_home.smart_copy(dst, indices),
        #             (v_home, False),
        #         ))
        # elif path == 1:  # Copy to CPU temporary workspace
        #     # shape: (s, b * n_head, head_dim)
        #     k_buf, v_buf = dst.next_attention_compute_workspace()
        #     indices = (slice(0, self.task.prompt_len + i - 1),
        #                slice(0, k_home.shape[1]))
        #     general_copy(k_buf, indices, k_home, indices)

        #     if self.policy.attn_sparsity >= 1.0:
        #         general_copy(v_buf, indices, v_home, indices)
        #         cache_read_buf.store(((k_buf, False), (v_buf, False)))
        #     else:
        #         cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        # elif path == 2:  # Copy to both GPU and CPU
        #     # The caches are stored on both GPU and other devices.
        #     # Compute attention on gpu for caches stored on gpu.
        #     # Compute attention on cpu for caches stored on cpu/disk.
        #     gpu_k_buf = k_home.data[0][0]
        #     gpu_v_buf = v_home.data[0][0]

        #     # shape: (s, b * n_head, head_dim)
        #     k_buf, v_buf = dst.next_attention_compute_workspace()
        #     indices = (slice(0, self.task.prompt_len + i - 1),
        #                slice(gpu_k_buf.shape[1], k_home.shape[1]))
        #     general_copy(k_buf, indices, k_home, indices)
        #     general_copy(v_buf, indices, v_home, indices)
        #     cache_read_buf.store((((gpu_k_buf, k_buf,), False),
        #                           ((gpu_v_buf, v_buf,), False)))
        #     assert self.policy.attn_sparsity >= 1.0
        # else:
        #     raise ValueError(f"Invalid path: {path}")
        
    def store_cache(self, 
                    cache_home:ValueHolder, 
                    cache_write_buf:ValueHolder, 
                    i:List[Union[int, float]]):
        # # shape: (s, b * n_head, head_dim)
        # k_home, v_home = cache_home.val
        # k_new, v_new = cache_write_buf.pop()

        # if i == self.task.gen_len - 1:  # last token, no need to store cache
        #     return

        # if i == 0:  # prefill
        #     indices = (slice(0, k_new.shape[0]),
        #                slice(0, k_new.shape[1]))
        # else:  # decoding
        #     pos = self.task.prompt_len + i
        #     indices = (slice(pos - k_new.shape[0], pos),
        #                slice(0, k_new.shape[1]))

        # general_copy(k_home, indices, k_new, None)
        # general_copy(v_home, indices, v_new, None)

        # shape of k_home/v_home: (batch_size, max_seq_len, num_kv_heads, head_dim)
        k_home, v_home = cache_home.val
        # k_new/v_new are the tensors to be written to the cache.
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # Prefill phase
            # k_new shape: (batch_size, prompt_len, num_kv_heads, head_dim)
            # We copy the entire prefill tensor into the beginning of the cache.
            prompt_len = k_new.shape[1]
            indices_dst = (slice(0, k_new.shape[0]), slice(0, prompt_len))
            general_copy(k_home, indices_dst, k_new, None)
            general_copy(v_home, indices_dst, v_new, None)
        else:  # Decoding phase
            # k_new shape: (batch_size, 1, num_kv_heads, head_dim)
            # We append the single new token at the correct position.
            pos = self.task.prompt_len + i -1
            indices_dst = (slice(0, k_new.shape[0]), slice(pos, pos + 1))
            general_copy(k_home, indices_dst, k_new, None)
            general_copy(v_home, indices_dst, v_new, None)


    # def forward(self, 
    #             hidden_states: TorchTensor, 
    #             attention_mask: torch.Tensor, 
    #             weights: List[TorchTensor], 
    #             freqs_cis: torch.Tensor, 
    #             start_pos: int, 
    #             is_prefill: bool, 
    #             past_kv: Tuple[TorchTensor, TorchTensor]):
    #     w_q, w_k, w_v, w_o = [w.data for (w, _) in weights]
        
    #     if is_prefill:
    #         attn_output = self.computation.mha(hidden_states.data, attention_mask, w_q, w_k, w_v, w_o, self.config, freqs_cis, self.k_cache, self.v_cache, start_pos)
    #         return TorchTensor.create_from_torch(attn_output, self.compute_device), None, None
    #     else:
    #         past_k, past_v = past_kv
    #         attn_output, new_k, new_v = self.computation.mha_gen(hidden_states.data, w_q, w_k, w_v, w_o, self.config, freqs_cis, past_k.data, past_v.data)
    #         return TorchTensor.create_from_torch(attn_output, self.compute_device), TorchTensor.create_from_torch(new_k, self.compute_device), TorchTensor.create_from_torch(new_v, self.compute_device)

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:List[Union[int, float]], 
                k:List[Union[int, float]]):
        donate = [False] * 9
        h, donate[0] = hidden.val, True
        
        if k == self.policy.num_gpu_batches - 1:
            (w_q, donate[2]), (w_k, donate[3]), (w_v, donate[4]), (w_o, donate[5]), (w_norm, donate[6]) = weight_read_buf.pop()
        else:
            (w_q, _), (w_k, _), (w_v, _), (w_o, _), (w_norm, _) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute_device)
            h, new_k_cache, new_v_cache = self.computation.mha(
                compute_device=self.compute_device, 
                hidden_states=h, 
                attention_mask=mask, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, w_norm=w_norm, 
                n_head=self.config.n_head, config=self.config, donate=donate, 
                compress_cache=self.policy.compress_cache, comp_config=self.policy.comp_cache_config, 
                )
            cache_write_buf.store((new_k_cache, new_v_cache))
        else: # decode
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[7]), (v_cache, donate[8]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.computation.mha_gen(
                compute_device=self.compute_device, 
                hidden_states=h, attention_mask=mask, 
                w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, w_norm=w_norm,
                n_head=self.config.n_head, config=self.config, 
                donate=donate, attn_sparsity=self.policy.attn_sparsity,compress_cache=self.policy.compress_cache, comp_config=self.policy.comp_cache_config, 
                k_cache=k_cache, v_cache=v_cache
                )
            cache_write_buf.store((new_k_cache, new_v_cache))
        
        hidden.val = h

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
            ((h,), "mlp_norm", p / self.weight_map["norm"][0], self.config.dtype),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        w_gate, w_up, w_down ,w_norm = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute_device
            weight_read_buf.store((
                w_gate.smart_copy(dst1),
                w_up.smart_copy(dst1),
                w_down.smart_copy(dst1),
                w_norm.smart_copy(dst2)
            ))

    # def forward(self, 
    #             hidden_states:TorchTensor, 
    #             weights:List[TorchTensor]):
    #     w_gate, w_up, w_down = [w.data for (w, _) in weights]
    #     output = self.computation.mlp(hidden_states.data, w_gate, w_up, w_down, self.config)
    #     return TorchTensor.create_from_torch(output, self.compute_device)
    
    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:List[Union[int, float]], 
                k:List[Union[int, float]]):
        donate = [False] * 5
        h, donate[0] = hidden.val, True
        
        if k == self.policy.num_gpu_batches - 1:
            (w_gate, donate[1]), (w_up, donate[2]), (w_down, donate[3]), (w_norm, donate[4]) = weight_read_buf.pop()
        else:
            (w_gate, _), (w_up, _), (w_down, _), (w_norm, _), = weight_read_buf.val
            
        h = self.computation.mlp(self.compute_device, h, w_gate, w_up, w_down, w_norm, self.config, donate)
        hidden.val = h


class LlamaTransformerLayer(BaseTransformerLayer):
    """Llama 的 Transformer 层。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map: dict, 
                 computation:LlamaModelComputation,  
                 freqs_cis_table: Optional[torch.Tensor]=None):
        super().__init__(config, env, policy, weight_map)
        self.computation = computation
        self.attention = LlamaSelfAttention(config, env, policy, weight_map['attention'], self.computation)
        self.mlp = LlamaMLP(config, env, policy, weight_map['mlp'], self.computation)

        self.freqs_cis_table = freqs_cis_table
        
        # self.attention_norm_weight = None
        # self.ffn_norm_weight = None
    def init_weight(self, weight_home, path):
        home_attn, home_mlp = ValueHolder(), ValueHolder()
        self.attention.init_weight(home_attn, path)
        self.mlp.init_weight(home_mlp, path)
        
        # p = Path(path)
        # h = self.config.hidden_size
        # spec_attn_norm = [((h,), "atten_norm", p / self.weight_map["attention"]["norm"][0], self.config.dtype)]
        # spec_ffn_norm = [((h,), "ffn_norm", p / self.weight_map["mlp"]["norm"][0], self.config.dtype)]
        # norm_weights = (
        #     init_weight_list(spec_attn_norm, self.policy, self.env)[0],
        #     init_weight_list(spec_ffn_norm, self.policy, self.env)[0]
        # )
        # home_norm.store(norm_weights)
        weight_home.store((home_attn, home_mlp))
    
    def load_weight(self, weight_home:ValueHolder, weight_read_buf:ValueHolder, k:List[Union[int, float]]):
        read_buf_attn, read_buf_mlp = ValueHolder(), ValueHolder()
        (home_attn, home_mlp) = weight_home.val
        
        self.attention.load_weight(home_attn, read_buf_attn, k)
        self.mlp.load_weight(home_mlp, read_buf_mlp, k)
        
        if k == 0:
            weight_read_buf.store((read_buf_attn, read_buf_mlp))
            # w_attn_norm, w_ffn_norm = home_norm.val
            # dst = self.compute_device
            # read_buf_norm.store((w_attn_norm.smart_copy(dst), w_ffn_norm.smart_copy(dst)))
            # weight_read_buf.store((read_buf_attn, read_buf_mlp, read_buf_norm))
    
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


    # def set_task(self, task):
    #     self.task = task
    
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
        # is_prefill = (i == 0)
        # donate = [False] * 2
        # start_pos = 0 if is_prefill else self.task.prompt_len + i - 1
        # seqlen = self.task.prompt_len if is_prefill else 1
        # freqs_cis = self.freqs_cis_table[start_pos : start_pos + seqlen]
        
        # h, residual = hidden.val.data, hidden.val.data
        # if k == self.policy.num_gpu_batches - 1:
        #     buf_attn, buf_mlp, buf_norm = weight_read_buf.pop()
        #     (w_attn_norm, donate[0]), (w_ffn_norm, donate[1]), = buf_norm.val
        # else:
        #     buf_attn, buf_mlp, buf_norm = weight_read_buf.val
        #     (w_attn_norm, _), (w_ffn_norm, _), = buf_norm.val
        
        # # w_attn_norm, w_ffn_norm = buf_norm.val
        # h_norm = self.computation.rms_norm(h, w_attn_norm.data)
        
        # causal_mask = None
        # if is_prefill:
        #     mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
        #     causal_mask = torch.triu(mask, diagonal=1)
        
        # past_kv = cache_read_buf.pop() if not is_prefill else None
        
        # attn_output, k_new, v_new = self.attention.forward(
        #     TorchTensor.create_from_torch(h_norm, self.compute_device), causal_mask, 
        #     buf_attn.val, freqs_cis, start_pos, is_prefill, past_kv
        # )
        # if not is_prefill:
        #     cache_write_buf.store((k_new, v_new))
        # h = residual + attn_output.data
        
        # residual = h
        # h_norm = self.computation.rms_norm(h, w_ffn_norm.data)
        # mlp_output = self.mlp.forward(TorchTensor.create_from_torch(h_norm, self.compute_device), buf_mlp.val)
        # h = residual + mlp_output.data
        
        # hidden.val.data = h

        # donate = [False] * 3

        if k == self.policy.num_gpu_batches - 1:
            read_buf_attn, read_buf_mlp = weight_read_buf.pop()
        else:
            read_buf_attn, read_buf_mlp = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf_attn, attention_mask,
                               cache_write_buf, i, k)
        self.mlp.forward(hidden, None, read_buf_mlp, attention_mask, None, i, k)


class LlamaModel(BaseModel):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 path:str):
        super().__init__(config, env, policy, weight_map=weight_map, path=path) 

        self.computation = LlamaModelComputation()
        self.start_pos = 0

        s = getattr(self.config, "max_seq_len", None) or getattr(self.config, "max_position_embeddings", None)
        # head_dim = config.hidden_size // config.n_head
        # inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float().cuda() / head_dim))
        # self.freqs_cis = self.computation.precompute_freqs_cis(head_dim, s * 2, inv_freq)

        self.layers.append(LlamaInputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))
        for layer_id in range(self.config.num_hidden_layers):
            self.layers.append(LlamaTransformerLayer(
                                                    self.config, self.env, self.policy, 
                                                    self.weight_map['layers'][layer_id], self.computation, 
                                                    # self.freqs_cis
                                                    ))
        
        self.layers.append(LlamaRMSNorm(self.config, self.env, self.policy, self.weight_map, self.computation))
        self.layers.append(LlamaOutputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))


        self.init_cache_area() # 初始化用于保存权重 激活 cache 的内存区域

        self.task = None

        # Precompute rotary embedding frequencies
        self.compute_device.rotary_emb_cis = self.computation.precompute_freqs_cis(
            self.config.hidden_size // self.config.n_head,
            s * 2,
            1.0 / (10000.0 ** (torch.arange(0, self.config.hidden_size // self.config.n_head, 2).float() / (self.config.hidden_size // self.config.n_head))).cuda()
        )
        self.init_all_weights(flexgen_weight_path=self.path)

    def init_all_weights(self, flexgen_weight_path):
        # self.weight_home = array_1d(self.num_layers, ValueHolder)
        for layer_id, layer in enumerate(self.layers):
            layer.init_weight(self.weight_home[layer_id], flexgen_weight_path)

    def update_attention_mask(self, i, k):
        
        is_prefill = i == 0
        # cur_seq_len = self.task.prompt_len if is_prefill else 1
        
        if not is_prefill:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return
        
        # if cur_seq_len > 1:
        #     mask = torch.full(
        #         (cur_seq_len, cur_seq_len), float("-inf")
        #     ).cuda()

        #     mask = torch.triu(mask, diagonal=1)

        #     # When performing key-value caching, we compute the attention scores
        #     # only for the new sequence. Thus, the matrix of scores is of size
        #     # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        #     # j > cache_len + i, since row i corresponds to token cache_len + i.
        #     mask = torch.hstack([
        #         torch.zeros((cur_seq_len, self.start_pos)).cuda(),
        #         mask
        #     ])  

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

        
