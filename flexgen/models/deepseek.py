from typing import Dict, List, Union, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from flexgen.utils import ValueHolder
from flexgen.pytorch_backend import TorchTensor, general_copy
from flexgen.models.base import BaseModel, BaseModelLayer
from flexgen.models.utils import ExecutionEnv, Policy


def rms_norm(x: torch.Tensor, norm_weight: torch.Tensor, eps: float = 1e-5):
    """RMS Normalization - LLaMA使用RMSNorm而不是LayerNorm"""
    def _norm(x, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        # 移除 clamp 操作
        rsqrt_var = torch.rsqrt(variance + eps)
        return x * rsqrt_var
        
    output = _norm(x.float(), eps)
    result = output * norm_weight.float()
    return result.type_as(x)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast to [1, 1, seq_len, dim // 2]
    # freqs_cis = freqs_cis.unsqueeze(1).to(xq_.device)
    freqs_cis = freqs_cis.to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out

def _compute_inv_freq(dim: int, theta: float = 10000.0, device="cpu"):
    """根据官方逻辑计算逆频率张量"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return freqs

def input_embed(compute_device, inputs, w_token, pad_token_id, donate):
    """输入嵌入层 - LLaMA只有token embedding，没有position embedding"""
    token_ids = inputs.data
    if donate[0]: inputs.delete()
    # token embedding
    token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
    return TorchTensor.create_from_torch(token_embed, compute_device)

def output_embed(compute_device, inputs, w_ln, w_token, donate,
                    do_sample, temperature):
    """输出嵌入层 - 使用RMSNorm"""
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
    
    
    return (TorchTensor.create_from_torch(ids, compute_device),
            TorchTensor.create_from_torch(logits, compute_device))


def mha(compute_device, inputs, attention_mask,
        # weights
        w_q,
        w_kv_a, w_kv_b, w_kv_a_ln,
        w_o, w_in_ln,
        # config
        n_head, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, 
        # other
        freqs_cis, donate):
    """Multi-Head Attention for the prefill phase, based on DeepSeek-V2's specific projections."""
    # Input norm
    residual = inputs.data
    hidden_states = rms_norm(inputs.data, w_in_ln.data)
    
    b, s, h = hidden_states.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    query_shape = (b, s, -1, qk_head_dim)
    key_shape = (b, s, -1, qk_nope_head_dim + v_head_dim)

    # q = w_q.data @ hidden_states
    q = F.linear(hidden_states, w_q.data)
    
    q = q.view(query_shape).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    # compressed_kv = w_kv_a.data(hidden_states)
    compressed_kv = F.linear(hidden_states, w_kv_a.data)
    k_nope, k_pe = torch.split(compressed_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_nope = rms_norm(k_nope, w_kv_a_ln.data)
    k_nope = F.linear(k_nope, w_kv_b.data).view(key_shape).transpose(1, 2)
    k_nope, value_states = torch.split(k_nope, [qk_nope_head_dim, v_head_dim], dim=-1)

    # k_pe = k_pe.view(b, 1, s, qk_rope_head_dim)
    # k_pe, q_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis)
    # k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
    k_pe = k_pe.view(b, 1, s, qk_rope_head_dim)
    q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis) # <- 变量顺序已修正
    k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
    query_states = torch.cat((q_nope, q_pe), dim=-1)
    key_states = torch.cat((k_nope, k_pe), dim=-1)

    k_for_cache = key_states.transpose(1,2).contiguous()
    v_for_cache = value_states.transpose(1,2).contiguous()

    
    # 注意力得分计算
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (qk_head_dim ** 0.5)
    # 遮罩
    mask_fill_value = torch.finfo(torch.float16).min
    if s > 1:
        causal_mask = torch.triu(torch.full((s, s), mask_fill_value, device=attn_weights.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights + causal_mask
    if attention_mask is not None:
        padding_mask = ~(attention_mask.data.bool()).view(b, 1, 1, s)
        attn_weights = attn_weights.masked_fill(padding_mask, mask_fill_value)
    # if attention_mask is not None:
    #     attn_weights = attn_weights + attention_mask.data
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(b, s, n_head * v_head_dim)
    
    # Output projection
    output = F.linear(attn_output, w_o.data) + residual

    k_for_cache = k_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_head, qk_head_dim)
    v_for_cache = v_for_cache.permute(1, 0, 2, 3).reshape(s, b * n_head, v_head_dim)

    if donate[0]: inputs.delete()
    if donate[1]: attention_mask.delete()

    k = TorchTensor.create_from_torch(k_for_cache, compute_device)
    v = TorchTensor.create_from_torch(v_for_cache, compute_device)

    return (TorchTensor.create_from_torch(output, compute_device), k, v)

def mha_gen(compute_device, inputs, attention_mask,
        # weights
        w_q,
        w_kv_a, w_kv_b, w_kv_a_ln,
        w_o, w_in_ln,
        # config
        n_head, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, 
        # cache
        k_cache, v_cache,
        # other
        freqs_cis, donate): 
    residual = inputs.data
    src_s = attention_mask.shape[1]
    hidden_states = rms_norm(inputs.data, w_in_ln.data)
    b, tgt_s, h = hidden_states.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    query_shape = (b, tgt_s, -1, qk_head_dim)
    key_shape = (b, tgt_s, -1, qk_nope_head_dim + v_head_dim)

    # q = w_q.data(hidden_states)
    q = F.linear(hidden_states, w_q.data)
    q = q.view(query_shape).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    compressed_kv = F.linear(hidden_states, w_kv_a.data)
    k_nope, k_pe = torch.split(compressed_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_nope = rms_norm(k_nope, w_kv_a_ln.data)
    k_nope = F.linear(k_nope, w_kv_b.data).view(key_shape).transpose(1, 2)
    k_nope, value_states = torch.split(k_nope, [qk_nope_head_dim, v_head_dim], dim=-1)

    k_pe = k_pe.view(b, 1, tgt_s, qk_rope_head_dim)
    q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis)
    k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
    query_states = torch.cat((q_nope, q_pe), dim=-1)
    key_states = torch.cat((k_nope, k_pe), dim=-1)

    # 5. 准备 k_new 和 v_new 用于 cache
    k_new_for_cache = key_states.permute(2, 0, 1, 3).contiguous().reshape(-1, b * n_head, qk_head_dim)
    v_new_for_cache = value_states.permute(2, 0, 1, 3).contiguous().reshape(-1, b * n_head, v_head_dim)
    
    k_cached = k_cache.data[:src_s-tgt_s]
    v_cached = v_cache.data[:src_s-tgt_s]
    
    k_all = torch.cat([k_cached, k_new_for_cache], dim=0) # shape: (src_s, b * n_kv_heads, head_dim)
    v_all = torch.cat([v_cached, v_new_for_cache], dim=0)

    # 直接从 k_all 张量获取其真实的序列长度
    actual_seq_len = k_all.shape[0]
    # 使用真实长度进行 view 操作，并为保证内存连续性添加 .contiguous()
    key_states = k_all.permute(1, 0, 2).contiguous().view(b, n_head, actual_seq_len, qk_head_dim)
    value_states = v_all.permute(1, 0, 2).contiguous().view(b, n_head, actual_seq_len, v_head_dim)

    # 6. 注意力得分
    scaling = qk_head_dim ** -0.5
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
    value = F.linear(attn_output, w_o.data) + residual

    if donate[0]: inputs.delete()
    if donate[1]: attention_mask.delete()

    k_new = TorchTensor.create_from_torch(k_new_for_cache, compute_device)
    v_new = TorchTensor.create_from_torch(v_new_for_cache, compute_device)

    return TorchTensor.create_from_torch(value, compute_device), k_new, v_new

def mlp(compute_device, inputs, w_norm, w_gate, w_up, w_down, donate):
    """MLP with SwiGLU activation - LLaMA特有的激活函数"""
    
    # Pre-normalization with RMSNorm
    residual = inputs.data
    out = rms_norm(inputs.data, w_norm.data)
    
    gate = F.linear(out, w_gate.data)
    up = F.linear(out, w_up.data)       
    silu_gate = F.silu(gate)  # SiLU activation
    
    intermediate = silu_gate * up  # Element-wise multiplication
    
    # Down projection
    out = F.linear(intermediate, w_down.data)
    
    out = out + residual  # Residual connection

    if donate[0]: inputs.delete()
    return TorchTensor.create_from_torch(out, compute_device)


def moegate(inputs, weights, topk_method, num_group, topk_group, num_experts, routed_scaling_factor, top_k):
    hidden_states = inputs.data
    batch_size, seq_len, hidden_dim = hidden_states.shape
    ### compute gating score
    hidden_states = hidden_states.view(-1, hidden_dim)
    logits = F.linear(hidden_states.type(torch.float32), weights.type(torch.float32), None)
    scores = logits.softmax(dim=-1, dtype=torch.float32)

    # select top-k experts
    # greedy method is used for DeepSeek-V2-Lite
    # group_limited_greedy for DeepSeek-V2 and DeepSeek-V2-Chat
    if topk_method == "greedy":
        topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    elif topk_method == "group_limited_greedy":
        group_scores = scores.view(batch_size * seq_len, num_group, -1).max(dim=-1).values  # [n, num_group]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, num_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, num_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(batch_size * seq_len, num_group, num_experts // num_group)
            .reshape(batch_size * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        topk_weight, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = topk_weight * routed_scaling_factor
    ### expert-level computation auxiliary loss
    return topk_idx, topk_weight

def moe(inputs, topk_ids, topk_weight, experts, n_routed_experts, ep_rank=0):
    hidden_states = inputs.data
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    indicies = topk_ids.view(-1).argsort()
    sorted_tokens = hidden_states[indicies // topk_ids.shape[1]]

    # Process experts
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            continue
        end_idx = start_idx + num_tokens
        expert = experts[i + ep_rank * n_routed_experts]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)

    # Reorder and combine outputs
    new_x = torch.empty_like(outs)
    new_x[indicies] = outs
    hidden_states = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return hidden_states

def moe_mlp(compute_device, inputs, 
            # experts
            shared_experts, routed_experts,
            # moegate
            weights, topk_method, num_group, topk_group, num_experts, routed_scaling_factor, top_k, 
            # moe
            experts, n_routed_experts,
            # other
            donate, ):
    # inputs, weights, topk_method, num_group, topk_group, num_experts, routed_scaling_factor, top_k
    # inputs, topk_ids, topk_weight, experts, n_routed_experts, ep_rank=0
    residuals = inputs.data
    orig_shape = residuals.shape
    topk_indices, topk_weights = moegate(residuals, routed_experts, topk_method, num_group, topk_group, num_experts, routed_scaling_factor, top_k)
    hidden_states = residuals.view(-1, residuals.shape[-1])
    hidden_states = moe(hidden_states, topk_indices, topk_weights, experts, n_routed_experts).view(*orig_shape)
    hidden_states = hidden_states + shared_experts(residuals)
    if donate[0]: inputs.delete()
    return TorchTensor.create_from_torch(hidden_states, compute_device)

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


def  init_weight_list_moe(state_dict: Dict[str, torch.Tensor],
                          weight_names_shared: List[str],
                          weight_names_routed: List[List[str]],
                          policy: Policy,
                          env: ExecutionEnv,
                          compute_device) -> Tuple[List[TorchTensor], List[List[TorchTensor]]]:
    
    dev_percents = [policy.w_disk_percent,
                    policy.w_cpu_percent,
                    policy.w_gpu_percent,
                    policy.w_cxl_percent]
    
    dev_choices = [env.disk,
                   env.cpu,
                   env.gpu,
                   env.cxl]
    
    shared_ret = []
    
    shared_weights = [state_dict[name] for name in weight_names_shared]
    shared_w_shapes = [list(w.shape) for w in shared_weights]
    shared_w_dtypes = [w.dtype for w in shared_weights]
    
    for w, s, d in zip(shared_weights, shared_w_shapes, shared_w_dtypes):
        sw = compute_device.allocate(s, d, pin_memory=False) # TODO pin_memory ?
        sw.load_from_torch(w)
        shared_ret.append(sw)
    
    routed_weights = [
        [state_dict[name] for name in weight_names_routed[i]]
        for i in range(len(weight_names_routed))
    ]
    routed_w_shapes = [
        [list(w.shape) for w in routed_weights[i]]
        for i in range(len(routed_weights))
    ]
    routed_w_dtypes = [
        [w.dtype for w in routed_weights[i]]
        for i in range(len(routed_weights))
    ]
    sizes = [
        np.sum([np.prod(shape) for shape in shape_list])
        for shape_list in routed_w_shapes
    ]
    sizes_cumsum = np.cumsum(sizes, dtype=np.int64)
    
    routed_ret = []
    for expert, (weight_list, shape_list, dtype_list, size) in enumerate(zip(routed_weights,
                                                                             routed_w_shapes,
                                                                             routed_w_dtypes,
                                                                             sizes)):
        mid_percent = (sizes_cumsum[expert] - size / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        
        weights: List[TorchTensor] = [
            home.allocate(shape, dtype, pin_memory=False) # TODO pin_memory ?
            for shape, dtype in zip(shape_list, dtype_list)
        ]
        for weight, w in zip(weights, weight_list):
            weight.load_from_torch(w)
            
        routed_ret.append(weights)
        
    return (shared_ret, routed_ret)

###############################################################################################################


class DeepSeekV2LiteModel(BaseModel):
    def __init__(self, 
                 pretrained_model_path: str,
                 env:ExecutionEnv, 
                 policy:Policy):
        
        self.layers = []
        
        loaded = BaseModel.load_from_pretrained_model(pretrained_model_path)
        self.state_dict, config, self.tokenizer = loaded
        
        # TODO DEBUG
        config['pad_token_id'] = config['eos_token_id']
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        super().__init__(config, env, policy)
        
        self.config.torch_dtype = 'torch.bfloat16'
        
        self.pretrained_model_path = pretrained_model_path
        self.env = env
        self.policy = policy

        # Build layers
        self.layers.append(DeepSeekV2LiteInputEmbed(self.state_dict,
                                                    self.config,
                                                    self.env,
                                                    self.policy))
        
        for layer_idx in range(self.config.num_hidden_layers):
            if self.policy.sep_layer:
                self.layers.append(DeepSeekV2LiteSelfAttention(state_dict=self.state_dict,
                                                               config=self.config,
                                                               env=self.env,
                                                               policy=self.policy,
                                                               layer_idx=layer_idx))
                if layer_idx == 0:
                    self.layers.append(DeepSeekV2LiteMLP(state_dict=self.state_dict,
                                                         config=self.config,
                                                         env=self.env,
                                                         policy=self.policy,
                                                         layer_idx=layer_idx))
                else:
                    self.layers.append(DeepSeekV2LiteMoE(state_dict=self.state_dict,
                                                         config=self.config,
                                                         env=self.env,
                                                         policy=self.policy,
                                                         layer_idx=layer_idx))
            else:
                raise NotImplementedError("Only sep_layer=True is supported for DeepSeek-V2-Lite currently.")

        self.layers.append(DeepSeekV2LiteOutputEmbed(self.state_dict,
                                                     self.config,
                                                     self.env,
                                                     self.policy))
        
        self.init_all_buffer()
        self.init_all_weights()
        
        # after init weights, free self.state_dict
        self.state_dict = None

    def model_bytes(self):
        return 0
    
    def cache_bytes(self, batch_size, seq_len):
        return 0

    def hidden_bytes(self, batch_size, seq_len):
        return 0


class DeepSeekV2LiteInputEmbed(BaseModelLayer):
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
        
        
class DeepSeekV2LiteSelfAttention(BaseModelLayer):
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

        self.inv_freq = _compute_inv_freq(
            dim=self.config.qk_rope_head_dim,
            theta=self.config.rope_theta,
            device=self.compute_device.device
        )

    def init_weight(self, weight_home: ValueHolder):
        weight_names = [
            "model.layers.{}.input_layernorm.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.q_proj.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.kv_a_layernorm.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.kv_b_proj.weight".format(self.layer_idx),
            "model.layers.{}.self_attn.o_proj.weight".format(self.layer_idx)
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
        if sub_batch_idx == 0:
            dst = self.compute_device
            (w_in_ln, w_q, w_kv_a, w_kv_a_ln, w_kv_b, w_o) = weight_home.val
            weight_read_buf.store((w_in_ln.smart_copy(dst),
                                   w_q.smart_copy(dst), 
                                   w_kv_a.smart_copy(dst),
                                   w_kv_a_ln.smart_copy(dst),
                                   w_kv_b.smart_copy(dst),
                                   w_o.smart_copy(dst)))
            
    
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
        
        kdim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        vdim = self.config.v_head_dim

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
        
        donate = [False] * 16
        h, donate[0] = hidden.val, True
        if sub_batch_idx == self.policy.num_gpu_batches - 1:
            ((w_in_ln, donate[2]), (w_q, donate[3]), (w_kv_a, donate[4]), 
             (w_kv_a_ln, donate[5]), (w_kv_b, donate[6]), (w_o, donate[7])) = weight_read_buf.pop()
        else:
            ((w_in_ln, _), (w_q, _), (w_kv_a, _), (w_kv_a_ln, _), (w_kv_b, _), (w_o, _)) = weight_read_buf.val
        
        
        seq_len = h.shape[1]
        start_pos = 0 if token_idx == 0 else self.task.prompt_len + token_idx - 1
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        # 1. 创建当前需要的位置ID
        position_ids = torch.arange(
            start_pos, start_pos + seq_len, dtype=torch.long, device=self.compute_device.device
        ).unsqueeze(0) # shape: [1, seq_len]

        # 2. 准备用于矩阵乘法的 inv_freq 和 position_ids
        # inv_freq shape: [qk_rope_head_dim / 2]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # shape: [1, qk_rope_head_dim / 2, 1]
        
        position_ids_expanded = position_ids[:, None, :].float()
        # shape: [1, 1, seq_len]

        # 3. 通过矩阵乘法计算频率
        # [1, dim/2, 1] @ [1, 1, seq_len] -> [1, dim/2, seq_len]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        # shape after transpose: [1, seq_len, dim/2]

        # 4. 转换为复数形式，并去掉多余的batch维度以匹配 apply_rotary_emb 的输入
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).squeeze(0)
        # shape: [seq_len, dim/2]
        
        # TODO DEBUG donate
        
        mask, donate[1] = attention_mask.val.smart_copy(self.compute_device)
        if token_idx == 0: # Prefill
            output, new_k, new_v = mha(self.compute_device, h, mask,
                                       w_q, w_kv_a, w_kv_b, w_kv_a_ln, w_o, w_in_ln,
                                       n_head=self.config.num_attention_heads,
                                       qk_nope_head_dim=self.config.qk_nope_head_dim,
                                       qk_rope_head_dim=self.config.qk_rope_head_dim,
                                       v_head_dim=self.config.v_head_dim,
                                       kv_lora_rank=self.config.kv_lora_rank,
                                       freqs_cis=freqs_cis,
                                       donate=donate)
        else: 
            (k_cache, donate[7]), (v_cache, donate[8]) = cache_read_buf.pop()
            output, new_k, new_v = mha_gen(compute_device=self.compute_device,
                                            inputs=h,
                                            attention_mask=mask,
                                            w_q=w_q,
                                            w_kv_a=w_kv_a,
                                            w_kv_b=w_kv_b,
                                            w_kv_a_ln=w_kv_a_ln,
                                            w_o=w_o,
                                            w_in_ln=w_in_ln,
                                            n_head=self.config.num_attention_heads,
                                            qk_nope_head_dim=self.config.qk_nope_head_dim,
                                            qk_rope_head_dim=self.config.qk_rope_head_dim,
                                            v_head_dim=self.config.v_head_dim,
                                            kv_lora_rank=self.config.kv_lora_rank,
                                            k_cache=k_cache,
                                            v_cache=v_cache,
                                            freqs_cis=freqs_cis,
                                            donate=donate)
        cache_write_buf.store((new_k, new_v))
        hidden.val = output

class DeepSeekV2LiteMLP(BaseModelLayer):
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
        assert self.layer_idx == 0, "Only the first layer is MLP in DeepSeek-V2-Lite."
        weight_names = [
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight"]
        weights = init_weight_list(state_dict=self.state_dict,
                                   weight_names=weight_names,
                                   policy=self.policy,
                                   env=self.env)
        weight_home.store(weights)
        
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    sub_batch_idx: int):
        w_norm, w_gate, w_up, w_down = weight_home.val
        if sub_batch_idx == 0:
            dst = self.compute_device
            weight_read_buf.store((w_norm.smart_copy(dst),
                                   w_gate.smart_copy(dst),
                                   w_up.smart_copy(dst),
                                   w_down.smart_copy(dst)))

    def forward(self, 
            hidden, 
            cache_read_buf: ValueHolder, 
            weight_read_buf: ValueHolder, 
            attention_mask: ValueHolder,
            cache_write_buf: ValueHolder, 
            token_idx: int,
            sub_batch_idx:int):

        donate = [False] * 5
        h = hidden.val
        donate[0] = True
        
        if sub_batch_idx == self.policy.num_gpu_batches - 1:
            (w_norm, donate[1]), (w_gate, donate[2]), (w_up, donate[3]), (w_down, donate[4]) = weight_read_buf.pop()
        else:
            (w_norm, _), (w_gate, _), (w_up, _), (w_down, _) = weight_read_buf.val

        h = mlp(compute_device=self.compute_device,
                inputs=h,
                w_norm=w_norm,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                donate=donate)
        
        hidden.val = h

class DeepSeekV2LiteMoE(BaseModelLayer):
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
        assert self.layer_idx > 0, "Only the first layer is MLP in DeepSeek-V2-Lite."
        weight_names_shared = [
            "model.layers.{}.post_attention_layernorm.weight".format(self.layer_idx),
            "model.layers.{}.mlp.gate.weight".format(self.layer_idx),
            "model.layers.{}.mlp.shared_experts.gate_proj.weight".format(self.layer_idx),
            "model.layers.{}.mlp.shared_experts.up_proj.weight".format(self.layer_idx),
            "model.layers.{}.mlp.shared_experts.down_proj.weight".format(self.layer_idx)
        ]
        
        weight_names_routed = []
        for e in range(64): # 64 experts
            weight_names_expert = [
                "model.layers.{}.mlp.experts.{}.gate_proj.weight".format(self.layer_idx, e),
                "model.layers.{}.mlp.experts.{}.up_proj.weight".format(self.layer_idx, e),
                "model.layers.{}.mlp.experts.{}.down_proj.weight".format(self.layer_idx, e),
            ]
            weight_names_routed.append(weight_names_expert)
        
        weights = init_weight_list_moe(state_dict=self.state_dict,
                                        weight_names_shared=weight_names_shared,
                                        weight_names_routed=weight_names_routed,
                                        policy=self.policy,
                                        env=self.env,
                                        compute_device=self.compute_device)
        weight_home.store(weights)
        
    def load_weight(self, 
                    weight_home: ValueHolder, 
                    weight_read_buf: ValueHolder, 
                    sub_batch_idx: int):
        _shared_weight, _routed_weight = weight_home.val
        dst = self.compute_device
        if sub_batch_idx == 0:
            shared_weight = [w.smart_copy(dst) for w in _shared_weight]
            
            routed_weight = [
                [w.smart_copy(dst) for w in expert_weights]
                for expert_weights in _routed_weight
            ]
            weight_read_buf.store((shared_weight, routed_weight))

    def gate(self,
             hidden_states: torch.Tensor,
             weight_gate: TorchTensor):
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, hidden_dim)
        logits = F.linear(hidden_states.type(torch.float32), weight_gate.data.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # select top-k experts
        # greedy method is used for DeepSeek-V2-Lite
        # group_limited_greedy for DeepSeek-V2 and DeepSeek-V2-Chat
        if self.config.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        else:
            raise NotImplementedError("Top-k method not implemented")

        topk_weight = topk_weight * self.config.routed_scaling_factor
        return topk_idx, topk_weight
    
    def moe(self,
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weight: torch.Tensor,
            experts: List[List[Tuple[TorchTensor, bool]]]):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        indicies = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[indicies // topk_ids.shape[1]]

        # Process experts
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            w_gate, w_up, w_down = experts[i]
            expert_out = self.mlp_moe(tokens_for_this_expert, w_gate[0], w_up[0], w_down[0])
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)

        # Reorder and combine outputs
        new_x = torch.empty_like(outs)
        new_x[indicies] = outs
        hidden_states = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return hidden_states

    def mlp_moe(self,
                hidden_states: torch.Tensor,
                w_gate: TorchTensor,
                w_up: TorchTensor,
                w_down: TorchTensor):
        gate = F.linear(hidden_states, w_gate.data)
        up = F.linear(hidden_states, w_up.data)       
        silu_gate = F.silu(gate)  # SiLU activation
        
        intermediate = silu_gate * up  # Element-wise multiplication
        
        # Down projection
        out = F.linear(intermediate, w_down.data)
        
        out = out + hidden_states  # Residual connection
        return out
    
    
    def forward(self, 
                hidden, 
                cache_read_buf: ValueHolder, 
                weight_read_buf: ValueHolder, 
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, 
                token_idx: int,
                sub_batch_idx:int):
        # Compute input embedding
        donate = [False] * 15
        h: TorchTensor = hidden.val
        donate[0] = True

        if sub_batch_idx == self.policy.num_gpu_batches - 1:
            weight_shared, weight_route_experts = weight_read_buf.pop()
        else:
            weight_shared, weight_route_experts = weight_read_buf.val
        
        w_ln, w_gate, w_shared_experts_gate, w_shared_experts_up, w_shared_experts_down = weight_shared
        
        residuals_without_ln = h.data
        hidden_states = rms_norm(residuals_without_ln, w_ln[0].data)

        residuals_with_ln = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states, w_gate[0])
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights, weight_route_experts).view(*orig_shape)

        hidden_states = hidden_states + self.mlp_moe(residuals_with_ln, w_shared_experts_gate[0], w_shared_experts_up[0], w_shared_experts_down[0])

        hidden_states = residuals_without_ln + hidden_states
        hidden.val = TorchTensor.create_from_torch(hidden_states, self.compute_device)

        
class DeepSeekV2LiteOutputEmbed(BaseModelLayer):
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