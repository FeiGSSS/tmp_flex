from typing import Dict, List, Union

import numpy as np

import torch
import torch.nn.functional as F

from flexgen.utils import ValueHolder
from flexgen.pytorch_backend import TorchTensor, general_copy
from flexgen.models.base import BaseModel, BaseModelLayer
from flexgen.models.utils import ExecutionEnv, Policy


################# Llama compute functions ####################

def input_embed(compute_device, inputs, w_token, pad_token_id, donate):
    token_ids = inputs.data
    if donate[0]: inputs.delete()
    token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
    return TorchTensor.create_from_torch(token_embed, compute_device)


def rms_norm(x: torch.Tensor, norm_weight: torch.Tensor, eps: float = 1e-5):
    orig_dtype = x.dtype
    x_float = x.to(torch.float32)
    var = x_float.mul(x_float).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    y = x_float * inv_rms * norm_weight.to(torch.float32)
    return y.to(orig_dtype)
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算RoPE的频率张量"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def mha(compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
        n_head, n_kv_heads, freqs_cis, donate):
    """
    mha 的最终调试版本，会打印每一步的中间结果。
    """
    b, s, h = inputs.shape
    head_dim = h // n_head
    

    # 1. 前置归一化
    residual = inputs.data
    hidden = rms_norm(inputs.data, w_ln.data)
    

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

    k = TorchTensor.create_from_torch(k, compute_device)
    v = TorchTensor.create_from_torch(v, compute_device)
    return TorchTensor.create_from_torch(value, compute_device), k, v

def mha_gen(compute_device, inputs, attention_mask, w_q, w_k, w_v, w_out, w_ln,
        n_head, n_kv_heads, k_cache, v_cache, freqs_cis, donate):
    """
    Multi-head attention (decoding phase) - Final corrected version for K/V cache shaping.
    """

    b, tgt_s, h = inputs.shape
    src_s = attention_mask.shape[1]
    head_dim = h // n_head

    # 1. 前置归一化
    residual = inputs.data
    hidden = rms_norm(inputs.data, w_ln.data)

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

    k_new = TorchTensor.create_from_torch(k_new_for_cache, compute_device)
    v_new = TorchTensor.create_from_torch(v_new_for_cache, compute_device)

    return TorchTensor.create_from_torch(value, compute_device), k_new, v_new

def mlp(compute_device, inputs, w_gate, w_up, w_down, w_ln, donate):
    """MLP with SwiGLU activation - LLaMA特有的激活函数"""

    b, s, h = inputs.shape
    
    # Pre-normalization with RMSNorm
    residual = inputs.data
    out = rms_norm(inputs.data, w_ln.data)

    
    gate = F.linear(out, w_gate.data)
    up = F.linear(out, w_up.data)       
    silu_gate = F.silu(gate)  # SiLU activation
    
    intermediate = silu_gate * up  # Element-wise multiplication
    
    # Down projection
    out = F.linear(intermediate, w_down.data)
    
    out = out + residual  # Residual connection

    if donate[0]: inputs.delete()
    return TorchTensor.create_from_torch(out, compute_device)

def output_embed(compute_device, inputs, w_ln, w_token, donate, do_sample, temperature):
    """输出嵌入层 - 使用RMSNorm"""
    # 保存输入隐藏状态数据
    
    # RMSNorm instead of LayerNorm
    hidden = rms_norm(inputs.data, w_ln.data)

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

###############################################################################################################
def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def  init_weight_list(state_dict: Dict[str, torch.Tensor],
                     weight_names: List[str],
                     policy: Policy,
                     env: ExecutionEnv) -> List[TorchTensor]:
    
    dev_percents = [policy.w_disk_percent,
                    policy.w_cpu_percent,
                    policy.w_gpu_percent,
                    policy.w_cxl_percent]
    
    dev_choices = [env.disk,
                   env.cpu,
                   env.gpu,
                   env.cxl]
    
    _weights = [state_dict[name] for name in weight_names]
    shapes = [list(w.shape) for w in _weights]
    dtypes = [w.dtype for w in _weights]

    sizes = [np.prod(spec) for spec in shapes]
    sizes_cumsum = np.cumsum(sizes)
    
    ret = []
    for i, (_weight, shape, dtype, size) in enumerate(zip(_weights, shapes, dtypes, sizes)):
        mid_percent = (sizes_cumsum[i] - size / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)

        pin_memory = True if len(shape) < 2 else policy.pin_weight
        
        weight:TorchTensor = home.allocate(shape, dtype, pin_memory=pin_memory)
        weight.load_from_torch(_weight)
        
        ret.append(weight)
        
    return ret


###############################################################################################################

class LLaMAModel(BaseModel):
    def __init__(self, 
                 pretrained_model_path: str,
                 env:ExecutionEnv, 
                 policy:Policy):
        
        self.layers = []

        loaded = BaseModel.load_from_pretrained_model(pretrained_model_path)
        self.state_dict, config, self.tokenizer = loaded
        
        config['pad_token_id'] = config['eos_token_id']
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        super().__init__(config, env, policy)
        
        self.config.n_head = self.config.num_key_value_heads
        self.config.input_dim = self.config.hidden_size
        self.config.torch_dtype = 'torch.float16'

        self.pretrained_model_path = pretrained_model_path
        self.env = env
        self.policy = policy
        
        self.layers.append(LLaMAInputEmbed(self.state_dict, self.config, self.env, self.policy))

        for layer_id in range(self.config.num_hidden_layers):
            if self.policy.sep_layer:
                self.layers.append(LLaMASelfAttention(self.state_dict,
                                                      self.config,
                                                      self.env,
                                                      self.policy,
                                                      layer_id))
                
                self.layers.append(LLaMAMLP(self.state_dict,
                                            self.config,
                                            self.env,
                                            self.policy,
                                            layer_id))
            else:
                raise NotImplementedError('Only support seperate layer for now.')
        
        self.layers.append(LLaMAOutputEmbed(self.state_dict, self.config, self.env, self.policy))

        self.init_all_buffer()
        self.init_all_weights()
        
        # after init weights, free self.state_dict
        self.state_dict = None
        
    def model_bytes(self):
        h = self.config.hidden_size
        return 	2 * (self.config.num_hidden_layers * (
                        # self-attention
                        h * (3 * h + 1) + h * (h + 1) +
                        # mlp
                        h * (4 * h + 1) + h * 4 * (h + 1) +
                        # layer norm
                        h * 4) +
                        # embedding
                        self.config.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.config.num_hidden_layers * self.config.hidden_size * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.config.hidden_size * 2

class LLaMAInputEmbed(BaseModelLayer):
    def __init__(self, 
                 state_dict: Dict[str, torch.Tensor],
                 config: Dict, 
                 env: ExecutionEnv, 
                 policy: Policy):
        self.state_dict = state_dict
        self.config = config
        self.env = env
        self.policy = policy
        
        self.compute_device = self.env.gpu
    
    def init_weight(self, weight_home: ValueHolder):
        weight_names = ["model.embed_tokens.weight"]
        weights = init_weight_list(state_dict=self.state_dict,
                                   weight_names=weight_names,
                                   policy=self.policy,
                                   env=self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    sub_batch_idx: int):
        w_token, = weight_home.val
        if sub_batch_idx == 0:
            weight_read_buf.store(w_token.smart_copy(self.compute_device))
    
    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                i: List[Union[int, float]], 
                sub_batch_idx: List[Union[int, float]]):

        # Compute input embedding
        donate = [False] * 2
        h, donate[0] = hidden.val, True

        if sub_batch_idx == self.policy.num_gpu_batches - 1:
            w_token, donate[1] = weight_read_buf.pop()
        else:
            w_token, _ = weight_read_buf.val

        h = input_embed(compute_device=self.compute_device,
                        inputs=h,
                        w_token=w_token,
                        pad_token_id=self.config.pad_token_id,
                        donate=donate)
        hidden.val = h

class LLaMASelfAttention(BaseModelLayer):
    def __init__(self, 
                 state_dict: Dict[str, torch.Tensor],
                 config: Dict, 
                 env: ExecutionEnv, 
                 policy: Policy,
                 layer_idx: int):
        
        self.state_dict = state_dict
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        
        self.compute_device = self.env.gpu
        
        # Precompute RoPE frequencies
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        max_seq_len = self.config.max_position_embeddings
        self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2).to(self.compute_device.device)


    def init_weight(self, weight_home: ValueHolder):
        weight_names = [
            "model.layers.{}.self_attn.q_proj.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.k_proj.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.v_proj.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.o_proj.weight".format(self.layer_idx),
            "model.layers.{}.input_layernorm.weight".format(self.layer_idx),

        ]
        weights = init_weight_list(state_dict=self.state_dict,
                                   weight_names=weight_names,
                                   policy=self.policy,
                                   env=self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    sub_batch_idx: int):
        w_q, w_k, w_v, w_out, w_ln = weight_home.val
        if sub_batch_idx == 0:
            dst = self.compute_device
            weight_read_buf.store((w_q.smart_copy(dst),
                                   w_k.smart_copy(dst),
                                   w_v.smart_copy(dst), 
                                   w_out.smart_copy(dst),
                                   w_ln.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home: ValueHolder):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        elif self.policy.cache_cxl_percent == 100:
            device = self.env.cxl
        else:
            raise NotImplementedError()

        task = getattr(self, 'task', None)
        assert task is not None, "Please set task before init cache."

        kdim = self.config.hidden_size // self.config.num_key_value_heads
        vdim = kdim

        cache = device.init_cache_one_gpu_batch(self.config, task, self.policy, kdim, vdim)
        cache_home.store(cache)
    
    def load_cache(self, 
                   cache_home: ValueHolder, 
                   cache_read_buf: ValueHolder, 
                   token_idx: int):
        if token_idx == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val
        indices = (slice(0, self.task.prompt_len + token_idx - 1),
                   slice(0, k_home.shape[1]))

        dst = self.compute_device
        cache_read_buf.store((
            k_home.smart_copy(dst, indices),
            v_home.smart_copy(dst, indices),
        ))
        
    def store_cache(self, 
                    cache_home: ValueHolder, 
                    cache_write_buf: ValueHolder, 
                    token_idx: int):
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        # last token, no need to store cache
        if token_idx == self.task.gen_len - 1: 
            return

        if token_idx == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + token_idx
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def forward(self, 
                hidden: ValueHolder, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder,
                token_idx: int,
                sub_batch_idx: int):
        
        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if sub_batch_idx == self.policy.num_gpu_batches - 1:
            ((w_q, donate[2]), (w_k, donate[3]), (w_v, donate[4]), 
             (w_out, donate[5]), (w_ln, donate[6])) = weight_read_buf.pop()
        else:
            (w_q, _), (w_k, _), (w_v, _), (w_out, _), (w_ln, _) = weight_read_buf.val

        seq_len = h.shape[1]
        start_pos = 0 if token_idx == 0 else self.task.prompt_len + token_idx - 1
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        mask, donate[1] = attention_mask.val.smart_copy(self.compute_device)
        if token_idx == 0:  # prefill
            h, new_k_cache, new_v_cache = mha(self.compute_device, h, mask, w_q, w_k, w_v, w_out, w_ln,
                                              self.config.num_attention_heads, self.config.num_key_value_heads,
                                              freqs_cis, donate)
        else:  # decoding
            (k_cache, donate[7]), (v_cache, donate[8]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = mha_gen(self.compute_device, h, mask, w_q, w_k, w_v, w_out, w_ln,
                                                  self.config.num_attention_heads, self.config.num_key_value_heads,
                                                  k_cache, v_cache, freqs_cis, donate)
        
        cache_write_buf.store((new_k_cache, new_v_cache))
        hidden.val = h

class LLaMAMLP(BaseModelLayer):
    def __init__(self, 
                 state_dict: Dict[str, torch.Tensor],
                 config: Dict, 
                 env: ExecutionEnv, 
                 policy: Policy,
                 layer_idx: int):
        self.state_dict = state_dict
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        
        self.compute_device = self.env.gpu

    def init_weight(self, weight_home: ValueHolder):
        weight_names = [
            "model.layers.{}.mlp.gate_proj.weight".format(self.layer_idx),
            "model.layers.{}.mlp.up_proj.weight".format(self.layer_idx),
            "model.layers.{}.mlp.down_proj.weight".format(self.layer_idx),
            "model.layers.{}.post_attention_layernorm.weight".format(self.layer_idx)
        ]
        weights = init_weight_list(state_dict=self.state_dict,
                                   weight_names=weight_names,
                                   policy=self.policy,
                                   env=self.env)
        weight_home.store(weights)

    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    sub_batch_idx: int):
        w_gate, w_up, w_down, w_ln = weight_home.val
        if sub_batch_idx == 0:
            dst = self.compute_device
            weight_read_buf.store((w_gate.smart_copy(dst),
                                   w_up.smart_copy(dst),
                                   w_down.smart_copy(dst),
                                   w_ln.smart_copy(dst)))
    
    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                token_idx: int,
                sub_batch_idx:int):
        
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if sub_batch_idx == self.policy.num_gpu_batches - 1:
            ((w_gate, donate[1]), (w_up, donate[2]), 
             (w_down, donate[3]), (w_ln, donate[4])) = weight_read_buf.pop()
        else:
            ((w_gate, _), (w_up, _), (w_down, _), (w_ln, _)) = weight_read_buf.val

        h = mlp(self.compute_device, h, w_gate, w_up, w_down, w_ln, donate)
        hidden.val = h

class LLaMAOutputEmbed(BaseModelLayer):
    def __init__(self, 
                 state_dict: Dict[str, torch.Tensor],
                 config: Dict, 
                 env: ExecutionEnv, 
                 policy: Policy):
        self.state_dict = state_dict
        self.config = config
        self.env = env
        self.policy = policy
        
        self.compute_device = self.env.gpu
        
    def init_weight(self, weight_home: ValueHolder):
        weight_names = [
            "model.norm.weight",
            "lm_head.weight"
        ]
        weights = init_weight_list(state_dict=self.state_dict,
                                   weight_names=weight_names,
                                   policy=self.policy,
                                   env=self.env)
        weight_home.store(weights)
    
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    sub_batch_idx: int):
        w_ln, w_token = weight_home.val
        if sub_batch_idx == 0:
            dst = self.compute_device
            weight_read_buf.store((w_ln.smart_copy(dst), w_token.smart_copy(dst)))

    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                token_idx: int,
                sub_batch_idx: int):

        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if sub_batch_idx == self.policy.num_gpu_batches - 1:
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        task = getattr(self, 'task', None)
        assert task is not None, "Please set task before forward."
        
        h, logits = output_embed(self.compute_device, h, w_ln, w_token, donate, task.do_sample, task.temperature)
        if task.logits:
            hidden.val = (h, logits)
        else:
            hidden.val = (h, None)