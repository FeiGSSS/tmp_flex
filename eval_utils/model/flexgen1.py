import argparse
import copy
import logging
import os, sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
from accelerate import find_executable_batch_size
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from flexllmgen.utils import (
    str2bool,
    ExecutionEnv,
    GB, 
    DUMMY_WEIGHT,
    project_decode_latency, write_benchmark_log
)

from flexllmgen.pytorch_backend import (
    TorchDevice,
    TorchDisk,
    TorchMixedDevice,
    TorchTensor, 
    DeviceType, 
    AsyncIOManager,
    fix_recursive_import)

from flexllmgen.opt_config import get_opt_config
from flexllmgen.compression import CompressionConfig
from flexllmgen.flex_opt import OptLM, Policy, add_parser_arguments
from flexllmgen.timer import timers
from flexllmgen.utils import (
    ExecutionEnv,
    Policy,
    ValueHolder,
    array_1d, array_2d, array_3d, array_4d,
)
from flexllmgen.utils import Task as flexllmgen_Task

fix_recursive_import()

dir_name = Path(__file__).parent.parent.parent.resolve()
if not str(dir_name) in sys.path:
    print(f"{dir_name} added to sys.path.")
    sys.path.append(str(dir_name))
print(sys.path)
import utils.utils
from utils.api.instance import Instance
from utils.api.model import TemplateLM
from utils.api.registry import register_model
from utils.model.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    stop_sequences_criteria,
)
eval_logger = logging.getLogger(__name__)

class OutputEmbed_loglikehood(OutputEmbed):
    def forward(self, 
                       hidden: TorchTensor, 
                       cache_read_buf: ValueHolder, 
                       weight_read_buf: ValueHolder, 
                       attention_mask: ValueHolder, 
                       cache_write_buf: ValueHolder, 
                       i: int, 
                       batch_idx: int) -> TorchTensor:
        if batch_idx == self.policy.num_batches - 1:
            (w_ln, w_ln_del), (b_ln, b_ln_del), (w_token, w_token_del) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        hidden_dim = hidden.val.shape[-1]
        inputs = hidden.val
        
        x = F.layer_norm(inputs.data, (hidden_dim,), weight=w_ln.data, bias=b_ln.data)
        inputs.delete()
        
        # output embedding
        if w_token.device.DeviceType == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
        logits = F.linear(x, w_token.data)
        # eval_logger.info(f"self._model_call: {logits} \n ========================")
        # eval_logger.info(f"logits.shape: {logits.shape}, logits dtype: {logits.dtype} \n ========================")
        # exit()

        if batch_idx == self.policy.num_batches - 1:
            if w_ln_del: w_ln_del.delete()
            if b_ln_del: b_ln_del.delete()
            if w_token_del: w_token_del.delete()
            
        return logits
    
    def init_weight(self, weight_home: ValueHolder, path:str)-> None:
        """Load weights to DISK/CPU/(GPU) according to the policy."""
        vocab_size = self.config.vocab_size
        hidden_size = self.config.input_dim
        dtype = self.config.dtype
        
        weight_specs = [
            # w_ln
            ((hidden_size,), dtype, os.path.join(path, "decoder.layer_norm.weight")),
            # b_ln
            ((hidden_size,), dtype, os.path.join(path, "decoder.layer_norm.bias")),
            # w_token
            ((vocab_size, hidden_size), dtype, os.path.join(path, "decoder.embed_tokens.weight")),
        ]
        
        weights = self.init_weight_list(weight_specs, self.policy, self.env)
        
        weight_home.store(weights)



class OptLM_eval(OptLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.layers_logits = copy.deepcopy(self.layers)
        self.layers_logits[-1] = OutputEmbed_loglikehood(config=self.config, env=self.env, policy=self.policy)
        self.num_layers_logits = len(self.layers_logits)

        # cache[j][k]
        self.cache_home_logits = array_2d(self.num_layers_logits, self.policy.num_batches, ValueHolder)
        self.cache_read_buf_logits = array_2d(self.num_layers_logits, self.policy.num_batches, ValueHolder)
        self.cache_write_buf_logits = array_2d(self.num_layers_logits, self.policy.num_batches, ValueHolder)
        # weight[j]
        self.weight_home_logits = array_1d(self.num_layers_logits, ValueHolder)
        self.weight_read_buf_logits = array_1d(self.num_layers_logits, ValueHolder)
        # attention_mask[k]
        self.attention_mask_logits = array_1d(self.policy.num_batches, ValueHolder)

        # 重新初始化这些 logits 相关的 weight_home
        # 确保每个 layer_logits 都使用其自己的 init_weight 方法来填充 weight_home_logits
        for j in range(self.num_layers_logits):
            expanded_path = os.path.abspath(os.path.expanduser(os.path.join(self.path, f"{self.config.name}-np")))
            check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
            if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
                eval_logger.info(f"Weight path {check_path} does not exist. Attempting to download or use dummy weights.")
                # 这里应该有处理下载或使用虚拟权重的逻辑，否则会报错
                # 例如： download_opt_weights(self.config.name, self.path)
                # 或抛出异常让用户处理
                raise Exception(f"Weight path {check_path} does not exist. Please ensure model weights are available.")

            self.weight_home_logits[j].clear() # 确保是空的ValueHolder
            self.layers_logits[j].init_weight(self.weight_home_logits[j], expanded_path)

                
        
    def forward_loglikelihood(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ):
        
        """
        执行单次前向传播以获取给定 input_ids 的 logits，与原 flexllmgen 提供的只生成 token 的 API 不同，需要重写
        原 flexllmgen 提供的只生成 token 的 API 可用到 generate_until
        新增 forward ，以满足 lm-evaluation-harness 中 loglikelihood 的计算需求。

        Args:
            input_ids (torch.Tensor): 输入 token IDs，形状 (batch_size, sequence_length)。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，形状 (batch_size, sequence_length)。

        Returns:
            torch.Tensor: Logits，形状 (batch_size, sequence_length, vocab_size)。
        """

        if self.num_batches != 1:
            eval_logger.error("OptLM_eval.forward 仅在当前实现中支持 num_batches=1 的策略。")
            raise NotImplementedError("OptLM_eval.forward does not support num_batches > 1 yet.")
            
        batch_size, seq_len = input_ids.shape
        batch_idx = 0 # 假设我们只处理第一个批次 (k=0)
        overlap = self.policy.overlap

        ## 参考原本的 OptLM.generate 函数
        with torch.no_grad():
            # 1. 设置 Task
            # 对于 loglikelihood，我们关注的是 prompt_len = seq_len, gen_len = 0
            # 因为我们不生成新的 token，只是计算给定序列的 logits
            # OptLM 的 task 需要 prompt_len 和 gen_len
            # 在这里，我们将整个 input_ids 视为 prompt，不进行生成

            task = flexllmgen_Task(
                inputs=input_ids.cpu().numpy().tolist(), # flexllmgen 的 Task 期望 list of lists
                prompt_len=seq_len,
                gen_len=1, # 不生成新的token
                cut_gen_len=1,
                do_sample=False,
                temperature=1.0,
                stop=None,
            )
            self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
            self.set_task_logits(task)


            # 2. 初始化 hidden states, cache, weight_read_buf, attention_mask
            # Loglikelihood 模式不需要 output_ids 和 stopped 状态
            # self.output_ids = ...
            # self.stopped = ...

            # 清空缓存相关，loglikelihood 不使用动态缓存
            for j in range(self.num_layers_logits):
                for k in range(self.policy.num_batches):
                    self.cache_home_logits[j][k].clear()
                    self.cache_read_buf_logits[j][k].clear()
                    self.cache_write_buf_logits[j][k].clear()

             # 清空 weight_read_buf_logits (在 load_weight_logits 中会填充)
            for j in range(self.num_layers_logits):
                self.weight_read_buf_logits[j].clear()

            # 清空 attention_mask_logits (在 update_attention_mask_logits 中会填充)
            for k in range(self.policy.num_batches):
                self.attention_mask_logits[k].clear()

            # hidden state buffer for current forward pass
            self.hidden_loglikelihood = array_3d(1, self.num_layers_logits, self.policy.num_batches, ValueHolder)

            for j in range(self.num_layers_logits):
                for k in range(self.policy.num_batches):
                    self.init_cache_logits(j, k)

            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

            # 初始化注意力掩码
            attention_compute = self.env.cpu
            if attention_mask is None:
                mask_np = (input_ids.cpu().numpy() != self.config.pad_token_id)
            else:
                mask_np = attention_mask.cpu().numpy().astype(bool)

            mask_val_holder = attention_compute.allocate((batch_size, seq_len), bool)
            mask_val_holder.load_from_np(mask_np)
            self.attention_mask_logits[batch_idx].store(mask_val_holder)

            # # 加载所有层的权重
            # for j in range(self.num_layers_logits):
            #     self.load_weight_logits(token_idx=0, layer_idx=j, batch_idx=batch_idx, overlap=True)
            # self.sync()  # 等待所有权重加载完成

            # 加载初始输入 embedding
            val_token_ids = self.env.cpu.allocate((batch_size, seq_len), np.int64)
            val_token_ids.load_from_np(input_ids.cpu().numpy().astype(np.int64))
            # 将 token IDs 存储到第一个隐藏状态 ValueHolder
            self.hidden_loglikelihood[0][0][batch_idx].store(val_token_ids)

            # 3. 逐层执行前向传播
            token_idx_for_forward = 0

            if not self.policy.overlap:
                # No overlap, easy to understand, suitable for debugging
                self.loglikelihood_normal()
            else:
                # Overlap I/O and compute
                if self.num_batches == 1:
                    self.loglikelihood_single_batch()
                else:
                    raise NotImplementedError("Only support num_batches=1 for now")
            # for j in range(self.num_layers_logits):
            #     self.load_weight_logits(0, j+1, 0)
            #     self.load_cache_logits(0, j+1, 0)
            #     self.load_hidden_logits(0, j, 0)
            #     self.compute_layer_logits(0, j, 0)
            #     self.store_cache_logits(0, j-1, 0)
            #     self.store_hidden_logits(0, j, 0)
            #     self.sync()
                # current_hidden_val_holder = self.hidden_loglikelihood[token_idx_for_forward][j][batch_idx]
                # current_weight_read_buf = self.weight_read_buf_logits[j]
                # current_attention_mask_val_holder = self.attention_mask_logits[batch_idx]

                # # Cache read/write buffers are passed, but for gen_len=0, they should not perform actual data movement
                # current_cache_read_buf = self.cache_read_buf_logits[j][batch_idx]
                # current_cache_write_buf = self.cache_write_buf_logits[j][batch_idx]

                # # 调用层的前向方法
                # self.layers_logits[j].forward(
                #     current_hidden_val_holder,
                #     current_cache_read_buf,
                #     current_weight_read_buf,
                #     current_attention_mask_val_holder,
                #     current_cache_write_buf,
                #     token_idx_for_forward,
                #     batch_idx
                # )

                # #释放上一层的 hidden，并为下一层准备输入
                # if j < self.num_layers_logits - 1:
                #     # 从当前层的 ValueHolder 中取出 TorchTensor，并移动到 CPU
                #     # 然后存储到下一层的 ValueHolder 中
                #     next_layer_input = self.hidden_loglikelihood[token_idx_for_forward][j][batch_idx].pop().move(self.env.cpu)
                #     self.hidden_loglikelihood[token_idx_for_forward][j+1][batch_idx].store(next_layer_input)
                # self.sync()  # 每层计算后同步，确保数据就绪

            # 4. 获取最终 logits
            final_hidden_val_holder = self.hidden_loglikelihood[token_idx_for_forward][self.num_layers_logits - 1][batch_idx]
            logits_tensor = final_hidden_val_holder.pop().data  # 提取 torch.Tensor

            # 清理所有层和批次的权重和缓存
            for j in range(self.num_layers_logits):
                self.delete_weight_logits(j, batch_idx)
                for k in range(self.policy.num_batches):
                    # loglikelihood 不会实际使用 cache，但为了兼容性，调用 delete_cache_logits 清理 ValueHolder
                    self.delete_cache_logits(j, k)
            self.env.cpu.del_attention_compute_workspace()

            return logits_tensor
    
    def load_weight_logits(self,
                           token_idx: int,
                           layer_idx: int,
                           batch_idx: int,
                           overlap: bool=True):
        # Handle corner cases for loglikelihood mode: token_idx should always be 0
        if layer_idx == self.num_layers_logits:
            layer_idx = 0
            token_idx += 1
            if token_idx == self.execute_gen_len: return # 在 loglikelihood 模式下，我们只处理 token_idx=0 的情况

        # Load from weight_home_logits to weight_read_buf_logits
        if overlap:
            self.load_weight_stream.submit(
                self.layers_logits[layer_idx].load_weight,
                self.weight_home_logits[layer_idx],
                self.weight_read_buf_logits[layer_idx],
                batch_idx
            )
        else:
            self.layers_logits[layer_idx].load_weight(self.weight_home_logits[layer_idx],
                                                      self.weight_read_buf_logits[layer_idx],
                                                      batch_idx)

    def delete_weight_logits(self, layer_idx:int, batch_idx:int):
        if batch_idx == 0: # 仅在第一个批次清理，因为权重通常跨批次共享
            if self.weight_home_logits[layer_idx].val: # 检查 ValueHolder 是否有值
                for x in self.weight_home_logits[layer_idx].pop():
                    if isinstance(x, ValueHolder):
                        if x.val: # 再次检查内部 ValueHolder 是否有值
                            for y in x.pop():
                                y.delete()
                    else:
                        x.delete()

    def init_cache_logits(self, layer_idx: int, batch_idx: int):
        """Initialize cache for a layer and a batch."""
        # 对于 loglikelihood，gen_len=0，所以实际不分配或使用缓存，但需要调用以匹配接口
        self.layers_logits[layer_idx].init_cache_one_batch(self.cache_home_logits[layer_idx][batch_idx])

    def load_cache_logits(self, token_idx, layer_idx, batch_idx, overlap=True):
        # 对于 loglikelihood，token_idx 始终为 0，且 gen_len=0，所以不涉及加载动态缓存
        pass

    def store_cache_logits(self, token_idx, layer_idx, batch_idx, overlap=True):
        # 对于 loglikelihood，token_idx 始终为 0，且 gen_len=0，所以不涉及存储动态缓存
        pass

    def delete_cache_logits(self, layer_idx: int, batch_idx: int):
        v = self.cache_home_logits[layer_idx][batch_idx].pop()
        if v:
            for x in v:
                x.delete()
    
    def compute_layer_logits(self, token_idx, layer_idx, batch_idx):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last batch
        # Clear the cache_read_buf
        # Run layer computation
        print(f"layer {layer_idx} \n ===================*******************")
        self.layers_logits[layer_idx].forward(self.hidden_loglikelihood[token_idx][layer_idx][batch_idx],
                                       self.cache_read_buf_logits[layer_idx][batch_idx],
                                       self.weight_read_buf_logits[layer_idx],
                                       self.attention_mask_logits[batch_idx],
                                       self.cache_write_buf_logits[layer_idx][batch_idx],
                                       token_idx,
                                       batch_idx)

    def load_hidden_logits(self, token_idx, layer_idx, batch_idx):
        # 这个函数在 forward_loglikelihood 中已被直接逻辑取代，因此可以安全地不在这里实现具体的加载逻辑
        # 或者，如果它是被其他地方调用的通用加载函数，则需要保留。
        # 但在 loglikelihood 模式下，input_ids 的加载和中间 hidden state 的传递逻辑都直接在 forward_loglikelihood 循环中。
        # 因此，这个函数可以安全地置空或移除。
        pass

    def store_hidden_logits(self, token_idx, layer_idx, batch_idx):
        # 这个函数在 forward_loglikelihood 中已被直接逻辑取代，因此可以安全地不在这里实现具体的存储逻辑
        # 且 loglikelihood 模式下不涉及 output_ids 和 stopped 状态的更新。
        # 因此，这个函数可以安全地置空或移除。
        pass

    # compute_layer 和 compute_layer_logits 在 forward_loglikelihood 中已被直接调用 self.layers_logits[j].forward 替代
    # 所以这些函数可以从 OptLM_eval 中移除，如果你确定它们只在此处被使用。
    # 否则，如果 OptLM 基类中仍有调用它们的逻辑，你需要决定如何处理，例如重写为空。
    # 但由于你已经创建了 layers_logits 并直接调用其 forward，这些中间的 compute_layer 函数是多余的。

    # update_attention_mask_logits 同样在 forward_loglikelihood 中被直接逻辑取代
    # 且 loglikelihood 模式下 mask 是静态的，不会动态扩展。
    def update_attention_mask_logits(self, token_idx, batch_idx):
        pass # 在 loglikelihood 模式下，mask 在 forward_loglikelihood 开始时一次性初始化，不需要动态更新

    def loglikelihood_normal(self, ):
        for k in range(self.num_batches):
            self.update_attention_mask_logits(0, k)
        for j in range(self.num_layers):
            for k in range(self.num_batches):
                self.load_weight_logits(0, j, k, overlap=False)

            for k in range(self.num_batches):
                self.load_cache_logits(0, j, k, overlap=False)
                self.load_hidden_logits(0, j, k)
                self.sync()
                self.compute_layer_logits(0, j, k)
                self.store_hidden_logits(0, j, k)
                self.store_cache_logits(0, j, k, overlap=False)
                self.sync()
    def loglikelihood_single_batch(self, ):
        for k in range(self.num_batches):
            self.load_weight_logits(0, 0, k)
        self.sync()
        # Generate
        for i in range(self.execute_gen_len):
            # self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight_logits(i, j+1, 0)
                self.load_cache_logits(i, j+1, 0)
                self.load_hidden_logits(i, j, 0)
                self.compute_layer_logits(i, j, 0)
                self.store_cache_logits(i, j-1, 0)
                self.store_hidden_logits(i, j, 0)
                self.sync()

            if self.task.stop and np.all(self.stopped):
                break
    def set_task_logits(self, task):
        self.task = task
        for l in self.layers_logits:
            l.set_task(task)

        
        


@register_model("flexllmgen", "flexllmgen")
class flexllmgen(TemplateLM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        args: Union[argparse.Namespace],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: str = "",
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        softmax_dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: Optional[int] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None, 
        **kwargs,
    ) -> None:
        super().__init__()
        # # TODO: take in an already-initialized transformers.PreTrainedModel

        # # args include three arguments, e.g. model, model_type, path, which may be confused
        # #      model is which opt model to use;
        # #      model_type is the class to instance model, like flexllmgen or HFLM; 
        # #      path is the path of the model

        if not isinstance(args.model_type, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            #  = args.model_type
            self._device = self._model.device
            self._config = self._model.config
            
        else:
            assert isinstance(args.device, str)
            assert isinstance(args.model_type, str)
            assert isinstance(args.gpu_batch_size, (int, str))

                # gpus = torch.cuda.device_count() # when use gpu, use this
                # 由于移除了 Accelerate 的分布式初始化，这里直接设置设备
            self._device = args.device
            eval_logger.info(f"{self._device}")

            revision = str(revision)  # cast to string if not already one

            # get self.config, which used in func self._get_backend and configure_pad_token(this func to configure self.tokenizer)
            self._get_config(
                args.model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                subfolder=subfolder,
            )

        # set up the configure to get OptLM instance
        num_prompts = args.num_gpu_batches * args.gpu_batch_size
        prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len
        self.cpu = TorchDevice("cpu")
        self.disk = TorchDisk(args.offload_dir)
        self.env = ExecutionEnv(cpu=self.cpu, disk=self.disk, mixed=TorchMixedDevice([self.cpu, self.disk]))
        self.policy = Policy(args.gpu_batch_size,
                    args.num_gpu_batches,
                    args.percent[0],
                    args.percent[1],
                    args.percent[2],
                    args.overlap,
                    args.sep_layer,
                    args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    
        assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

        self.opt_config = get_opt_config(args.model)
        cache_size = self.opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
        hidden_size = self.opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
        print(f"model size: {self.opt_config.model_bytes()/GB:.3f} GB, "
            f"cache size: {cache_size/GB:.3f} GB, "
            f"hidden size (prefill): {hidden_size/GB:.3f} GB")

        print("init weight...")
        self._model = OptLM_eval(self.opt_config, self.env, args.path, self.policy)

            # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.path, padding_side="left")
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
            
        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        self.pretrained = args.path
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        self.softmax_dtype = (
            get_dtype(softmax_dtype) if softmax_dtype is not None else None
        )

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        # 简化分布式设置，假设单进程或由 model_load 处理 "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
        self._rank = 0
        self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.
        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder)) model type to be used.
        sets `self.AUTO_MODEL_CLASS` appropriately if not already set.

        **If not calling flexllmgen.__init__() or flexllmgen._get_backend() within a subclass of flexllmgen,
        user must set `self.backend` to be either "causal" or "seq2seq" manually!**
        """

        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.backend = backend
            elif backend == "seq2seq":
                self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.backend = "seq2seq"
                eval_logger.debug(f"Using model type '{self.backend}'")
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
                eval_logger.debug(f"Using model type '{self.backend}'")
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                        "Setting backend to causal"
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to assuming AutoModelForCausalLM
                self.backend = "causal"
                eval_logger.info(
                    f"Model type cannot be determined. Using default model type '{self.backend}'"
                )

        if self.AUTO_MODEL_CLASS is None:
            if self.backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif self.backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    # _get_config需要重写
    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
        subfolder: str = "",
    ) -> None:
        """Return the model config for HuggingFace models"""
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            gguf_file=gguf_file,
            subfolder=subfolder,
        )


    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            if self.backend == "seq2seq":
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length),
                ).long()
                test_batch = torch.ones((batch_size, length),).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length),
                ).long()
            for _ in range(5):
                out = F.log_softmax(  # noqa: F841
                    self._model_call(test_batch, **call_kwargs),
                    dim=-1,
                    dtype=self.softmax_dtype,
                )

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        clear_torch_cache()
        return batch_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            logits = self.model.forward_loglikelihood(inps, attention_mask=attn_mask)
            return logits
        
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS in (
                    transformers.AutoModelForCausalLM,
                    transformers.AutoModelForVision2Seq,
                )
                return self.model(inps).logits

    def _model_generate_old(self, context, max_length, stop, **generation_kwargs):
        # 设置温度参数，默认为0.0
        # temperature = 0.0 if not set
        # 如果 do_sample 为 false 且温度等于0.0，则移除温度参数，避免HF警告
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)

        # 获取 do_sample 参数，默认为 None
        do_sample = generation_kwargs.get("do_sample", None)

        # 如果温度参数为0.0且 do_sample 未设置，则使用贪心解码策略
        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        # 如果 do_sample 为 False 且温度参数为0.0，则移除温度参数
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        # 构建停止条件
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        # 调用模型生成文本
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # 这里的 context 是 PyTorch Tensor
        # OptLM.generate 期望的是 NumPy 数组或 List[List[int]]
        # 并且 max_length 对应 max_new_tokens
        inputs_np = context.cpu().numpy().tolist() # 转换为 List[List[int]]

        # 提取 OptLM.generate 所需的参数
        max_new_tokens = max_length - context.shape[1] # max_length 是总长度，需要减去 prompt 长度
        if max_new_tokens <= 0:
            eval_logger.warning(f"max_new_tokens is non-positive ({max_new_tokens}). Setting to 1.")
            max_new_tokens = 1

        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # OptLM 的 stop 参数是单个 token ID，而这里 stop 是一个列表
        # 需要找到第一个有效的 stop token ID
        optlm_stop_token_id = None
        if stop:
            for s_item in stop:
                if isinstance(s_item, int):
                    optlm_stop_token_id = s_item
                    break
                elif isinstance(s_item, str) and len(s_item) == 1: # 假设单字符 stop sequence
                    encoded_stop = self.tokenizer.encode(s_item, add_special_tokens=False)
                    if encoded_stop:
                        optlm_stop_token_id = encoded_stop[0]
                        break
            if optlm_stop_token_id is None:
                eval_logger.warning(f"Could not find a suitable stop token ID from {stop}. Generation will proceed without a specific stop token.")

        # OptLM.generate 还需要 cut_gen_len，但 lm_eval 没有直接提供
        # 可以从 args 中获取，或者设置为 None
        cut_gen_len = generation_kwargs.get("cut_gen_len", None) # 尝试从 kwargs 获取

        # 调用 OptLM 实例的 generate 方法
        # self.model 此时是 OptLM 的实例
        generated_ids_np = self.model.generate(
            inputs=inputs_np,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=generation_kwargs.get("temperature", 1.0),
            stop=optlm_stop_token_id, # 传递给 OptLM 的 stop 参数
            cut_gen_len=cut_gen_len,
            verbose=0 # 可以根据需要调整
        )

        # OptLM.generate 返回的是 NumPy 数组，需要转换为 PyTorch Tensor
        # 并且只返回生成的 continuation 部分
        # generated_ids_np 包含 prompt + generated tokens
        # 我们需要提取生成的 tokens 部分
        
        # 假设 generated_ids_np 是 (batch_size, prompt_len + gen_len)
        # 那么生成的 tokens 从 prompt_len 开始
        generated_only_np = generated_ids_np[:, context.shape[1]:]
        
        # 将 NumPy 数组转换为 PyTorch Tensor，并移动到正确设备
        generated_tokens_tensor = torch.tensor(generated_only_np, dtype=torch.long, )
        
        # _model_generate 返回的是生成的 token ID 序列，而不是 logits
        # flexllmgen.generate_until 期望 _model_generate 返回的是生成的 token ID 序列
        # 因此，直接返回这个 tensor 即可
        return generated_tokens_tensor

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        if self.backend == "causal":
            assert contlen and inplen, (
                "Must pass input len and cont. len to select scored logits for causal LM"
            )
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self.backend == "seq2seq":
            assert contlen and not inplen, (
                "Selecting scored logits for Seq2SeqLM requires only cont. len"
            )
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    utils.utils.make_disjoint_window,
                    utils.utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls = []
        batch_size = adaptive_batch_size or self.batch_size
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
                override_bs=len(batch_windows),
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.backend == "causal" and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.backend == "causal":
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        eval_logger.warning(
                            f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                            f"exceeds model's maximum length ({self.max_length}). "
                            f"Truncating {total_length - self.max_length + 1} tokens from the left."
                        )
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                    )
                    (inplen,) = inp.shape
                elif self.backend == "seq2seq":
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.backend == "causal":
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.backend == "seq2seq":
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            # eval_logger.info(f"batched_inps are: {batched_inps} \n ========================")
            # eval_logger.info(f"call_kwargs are: {call_kwargs} \n ========================")
            # exit()
            eval_logger.info(f"dtype is: {self.softmax_dtype} \n ========================")
            eval_logger.info(f"self._model_call: {self._model_call(batched_inps, **call_kwargs)} \n ========================")
            eval_logger.info(f"logits.shape: {self._model_call(batched_inps, **call_kwargs).shape}, self._model_call dtype: {self._model_call(batched_inps, **call_kwargs).dtype} \n ========================")
            exit()
            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.backend == "causal"
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long,
                    ).unsqueeze(0)  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (
                        greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks
                    ).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
                )
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            # context_enc = context_enc.to(self.device)
            # attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res


    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
        }
        return model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    args.model = "facebook/opt-125m"
    args.path = "/shared/model/opt/opt-125m/"
    flexllmgen = flexllmgen(args, )