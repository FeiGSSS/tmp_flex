from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Union, List

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

class LLaMAModelComputation:
    def input_embed(self, compute_device, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, compute_device)

    def output_embed(self, compute_device, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
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

    def _attention_weights(self, compute_device, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, compute_device, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(compute_device, q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, compute_device, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(compute_device, q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = compute_device.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, compute_device, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(compute_device, q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(compute_device, q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mha(self, compute_device, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=compute_device.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = compute_device.compressed_device.compress(k, comp_config)
            v = compute_device.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, compute_device)
            v = TorchTensor.create_from_torch(v, compute_device)

        return TorchTensor.create_from_torch(value, compute_device), k, v

    def mha_gen(self, compute_device, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(compute_device, q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(compute_device, q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(compute_device, q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(compute_device, q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(compute_device, q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = compute_device.compressed_device.compress(k_new, comp_config)
            v_new = compute_device.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, compute_device)
            v_new = TorchTensor.create_from_torch(v_new, compute_device)

        return TorchTensor.create_from_torch(value, compute_device), k_new, v_new

    def mlp(self, compute_device, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, compute_device)

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
                 computation:LLaMAModelComputation):
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
            dst = self.torch_device
            weight_read_buf.store((w_norm.smart_copy(dst),))

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i: List[Union[int, float]], 
                k: List[Union[int, float]]):
        # TODO: 实现 RMSNorm 的前向计算
        pass


class LlamaInputEmbed(BaseModelLayer):
    """Llama 的输入嵌入层，只包含词嵌入。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LLaMAModelComputation):
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
            weight_read_buf.store((w_token.smart_copy(dst),))
    
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
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute_device)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.computation.input_embed(self.compute_device, h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h

class LlamaOutputEmbed(BaseModelLayer):
    """Llama 的输出层，将最终的 hidden states 映射到词汇表。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LLaMAModelComputation):
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
            weight_read_buf.store((w_lm_head.smart_copy(dst),))

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
        # TODO: 实现输出映射的前向计算
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        h, logits = self.computation.output_embed(self.compute_device, h, w_ln, b_ln, w_token, donate,
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
                 computation:LLaMAModelComputation):
        super().__init__(config, env, policy, weight_map=weight_map)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        self.computation = computation # 待实现

    def init_weight(self, 
                    weight_home:ValueHolder, 
                    converted_path: str):
        h = self.config.hidden_size
        p = Path(converted_path)
        
        weight_specs = [
            ((h, h), "atten_q_proj", p / self.weight_map["q_proj"][0], self.config.dtype),
            ((h, h), "atten_k_proj", p / self.weight_map["k_proj"][0], self.config.dtype),
            ((h, h), "atten_v_proj", p / self.weight_map["v_proj"][0], self.config.dtype),
            ((h, h), "atten_o_proj", p / self.weight_map["o_proj"][0], self.config.dtype),
            ((h,), "atten_norm", p / self.weight_map["norm"][0], self.config.dtype),
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

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

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
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:List[Union[int, float]], 
                k:List[Union[int, float]]):
        # TODO: 实现 Llama Attention 的前向计算 (包括 RoPE)
        pass

class LlamaMLP(BaseModelLayer):
    """Llama 的 MLP 层，使用 SwiGLU。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 computation:LLaMAModelComputation):
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
        w_gate, w_up, w_down, w_norm = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute_device
            weight_read_buf.store((
                w_gate.smart_copy(dst1),
                w_up.smart_copy(dst1),
                w_down.smart_copy(dst1),
                w_norm.smart_copy(dst2)
            ))

    def forward(self, 
                hidden, 
                cache_read_buf:ValueHolder, 
                weight_read_buf:ValueHolder, 
                attention_mask:ValueHolder,
                cache_write_buf:ValueHolder, 
                i:List[Union[int, float]], 
                k:List[Union[int, float]]):
        # TODO: 实现 Llama MLP (SwiGLU) 的前向计算
        pass


class LlamaTransformerLayer(BaseTransformerLayer):
    """Llama 的 Transformer 层。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map: dict, 
                 computation:LLaMAModelComputation):
        super().__init__(config, env, policy, weight_map)
        self.computation = computation
        self.attention = LlamaSelfAttention(config, env, policy, weight_map['attention'], self.computation)
        self.mlp = LlamaMLP(config, env, policy, weight_map['mlp'], self.computation)

    def init_weight(self, weight_home, path):
        # TODO: 实现组合权重初始化
        pass
    
    def load_weight(self, weight_home:ValueHolder, weight_read_buf:ValueHolder, k:List[Union[int, float]]):
        # TODO: 实现组合权重加载
        pass

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
        pass


class LlamaModel(BaseModel):
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 path:str):
        super().__init__(config, env, policy, weight_map=weight_map, path=path) 

        self.computation = LLaMAModelComputation()
        self.layers.append(LlamaInputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))
        for layer_id in range(self.config.num_hidden_layers):
            self.layers.append(LlamaTransformerLayer(self.config, self.env, self.policy, self.weight_map['layers'][layer_id], self.computation))
        
        self.layers.append(LlamaRMSNorm(self.config, self.env, self.policy, self.weight_map, self.computation))
        self.layers.append(LlamaOutputEmbed(self.config, self.env, self.policy, self.weight_map, self.computation))


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
        # self.weight_home = array_1d(self.num_layers, ValueHolder)
        for layer_id, layer in enumerate(self.layers):
            layer.init_weight(self.weight_home[layer_id], flexgen_weight_path)