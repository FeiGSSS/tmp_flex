import os
import json
from typing import Dict

#######################
import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer

from flexgen.utils import (ValueHolder, array_1d, array_2d, array_3d, str_to_dtype)
from flexgen.timer import timers
from flexgen.models.utils import ExecutionEnv, Policy, Task
from flexgen.pytorch_backend import TorchTensor
from types import SimpleNamespace

def flatten(xs):
    for x in xs:
        if isinstance(x, (list, tuple)):
            yield from flatten(x)
        else:
            yield x

class BaseModelLayer:
    def __init__(self):
        pass    
    
    def set_task(self, task):
        self.task = task

    def init_cache_one_gpu_batch(self, cache_home):
        pass  

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  

    def store_cache(self, cache_home, cache_write_buf, i):
        pass

class BaseModel:
    def __init__(self, 
                 config:Dict, 
                 env:ExecutionEnv, 
                 policy:Policy):
        
        self.config = SimpleNamespace(**config)
        self.env = env
        self.policy = policy
        
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        elif self.policy.act_cxl_percent == 100:
            self.act_home = self.env.cxl
        else:
            raise NotImplementedError("Activation memory placement policy not supported.")
        
        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        
    @classmethod
    def load_from_pretrained_model(cls, pretrained_model_path: str) -> Dict[str, torch.Tensor]:
        """Load model weights from a pretrained model.

        Args:
            pretrained_model_path (str): Path to the pretrained model.

        Raises:
            KeyError: If a required parameter is missing from the pretrained model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to their tensor values.
        """
        index_file = os.path.join(pretrained_model_path, "model.safetensors.index.json")
        assert os.path.isfile(index_file), f"缺少索引文件: {index_file}"
        with open(index_file, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map")
        
        # shard -> [param,...]
        shard_to_params = {}
        for pname, shard in weight_map.items():
            shard_to_params.setdefault(shard, []).append(pname)
            
        state_dict = {}
        for shard_name, params in shard_to_params.items():
            shard_path = os.path.join(pretrained_model_path, shard_name)
            shard_sd = safe_load_file(shard_path, device="cpu")
            for p in params:
                if p not in shard_sd:
                    raise KeyError(f"{p} 不在分片 {shard_name}")
                t = shard_sd[p]
                state_dict[p] = t
        
        
        config_file = os.path.join(pretrained_model_path, "config.json")
        assert os.path.isfile(config_file), f"缺少配置文件: {config_file}"
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        tok = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            use_fast=False,
            trust_remote_code=False,
            padding_side="left"
        )
        
        return state_dict, config_data, tok
        
    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)
            
    @property
    def num_gpu_batches(self, ):
        return self.policy.num_gpu_batches
    
    @property
    def num_layers(self, ):
        return len(self.layers)
    
    def init_all_buffer(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        # cache[j][k]
        self.cache_home = array_2d(self.num_layers, self.num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(self.num_layers, self.num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(self.num_layers, self.num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(self.num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(self.num_gpu_batches, ValueHolder)
        
    def init_all_weights(self):
        for layer_id, layer in enumerate(self.layers):
            layer.init_weight(self.weight_home[layer_id])
    
    def load_weight(self,  
                    token_id: int,
                    layer_id: int,
                    sub_batch_idx: int,
                    overlap:bool = True):
        # Handle corner cases
        if layer_id == self.num_layers:
            layer_id = 0
            token_id += 1
            if token_id == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[layer_id].load_weight(self.weight_home[layer_id],
                                                  self.weight_read_buf[layer_id],
                                                  sub_batch_idx)
        else:
            self.layers[layer_id].load_weight(self.weight_home[layer_id],
                                              self.weight_read_buf[layer_id],
                                              sub_batch_idx)
    
    def delete_all_weights(self):
        for j in range(self.num_layers):
            if self.weight_home[j]:
                for x in self.weight_home[j].pop():
                    if isinstance(x, TorchTensor):
                        x.delete()
                    else:
                        x = flatten(x)
                        for xx in x:
                            xx.delete()

    def init_cache(self, 
                   layer_id: int,
                   sub_batch_idx: int):
        self.layers[layer_id].init_cache_one_gpu_batch(self.cache_home[layer_id][sub_batch_idx])

    def load_cache(self,  
                   token_idx: int,
                   layer_id: int,
                   sub_batch_idx: int,
                   overlap:bool =True):
        # Handle corner cases
        if token_idx == 0:  # prefill, no cache
            return
        if sub_batch_idx == self.num_gpu_batches:
            sub_batch_idx = 0
            layer_id += 1
        if layer_id == self.num_layers:
            layer_id = 0
            token_idx += 1
            if token_idx == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[layer_id].load_cache(self.cache_home[layer_id][sub_batch_idx],
                                                 self.cache_read_buf[layer_id][sub_batch_idx],
                                                 token_idx)
        else:
            self.layers[layer_id].load_cache(self.cache_home[layer_id][sub_batch_idx],
                                             self.cache_read_buf[layer_id][sub_batch_idx],
                                             token_idx)

    def store_cache(self,  
                    token_idx: int,
                    layer_idx: int,
                    sub_batch_idx: int,
                    overlap:bool=True):
        # Handle corner cases
        if sub_batch_idx == -1:
            sub_batch_idx = self.num_gpu_batches - 1
            layer_idx -= 1
        if layer_idx == -1:
            layer_idx = self.num_layers - 1
            token_idx -= 1
            if token_idx == -1:
                return
        if token_idx == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[layer_idx][sub_batch_idx].pop()
            return

        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[layer_idx].store_cache(self.cache_home[layer_idx][sub_batch_idx],
                                                   self.cache_write_buf[layer_idx][sub_batch_idx],
                                                   token_idx)
        else:
            self.layers[layer_idx].store_cache(self.cache_home[layer_idx][sub_batch_idx],
                                               self.cache_write_buf[layer_idx][sub_batch_idx],
                                               token_idx)

    def delete_cache(self,
                     layer_idx: int,
                     sub_batch_idx: int):
        kv = self.cache_home[layer_idx][sub_batch_idx].pop()
        if kv:
            for x in kv:
                x.delete()

    def load_hidden(self, 
                    token_idx: int,
                    layer_idx: int,
                    sub_batch_idx: int):
        # Handle corner cases
        if sub_batch_idx == self.num_gpu_batches:
            sub_batch_idx = 0
            layer_idx += 1
        
        if layer_idx == self.num_layers:
            layer_idx = 0
            token_idx += 1
            if token_idx == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[layer_idx].compute_device
        if layer_idx == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = sub_batch_idx * gpu_batch_size, (sub_batch_idx + 1) * gpu_batch_size
            if token_idx == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), self.task.inputs[0].dtype)
                val.load_from_torch(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + token_idx
                val = dst.allocate((gpu_batch_size, 1), self.task.inputs[0].dtype)
                val.load_from_torch(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[token_idx][layer_idx-1][sub_batch_idx].pop().move(dst)
        self.hidden[token_idx][layer_idx][sub_batch_idx].store(val)

    def store_hidden(self, token_idx, layer_idx, sub_batch_idx):
        # Handle corner cases
        if sub_batch_idx == -1:
            sub_batch_idx = self.num_gpu_batches - 1
            layer_idx -= 1
        if layer_idx == -1:
            layer_idx = self.num_layers - 1
            token_idx -= 1
            if token_idx == -1:
                return

        # Store to hidden states buffers
        if layer_idx == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = sub_batch_idx * gpu_batch_size, (sub_batch_idx + 1) * gpu_batch_size

            ids, logits = self.hidden[token_idx][layer_idx][sub_batch_idx].pop()
            ids = ids.data.detach()
            if logits is not None:
                logits = logits.data.detach()
                self.logits[left:right] = logits
            
            pos = self.task.prompt_len + token_idx

            if self.task.stop:
                stopped = self.stopped[left:right]
                # self.output_ids[left:right, pos:pos+1] = np.where(stopped, self.config.pad_token_id, ids)
                # stopped[:] = np.logical_or(stopped, ids == self.task.stop)
                stopped = stopped.to(ids.device)
                pad_filled = ids.masked_fill(stopped, self.config.pad_token_id)
                self.output_ids[left:right, pos:pos+1] = pad_filled
                self.stopped[left:right] = torch.logical_or(stopped, ids == self.task.stop).to(self.stopped.device)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:
            x = self.hidden[token_idx][layer_idx][sub_batch_idx]
            # if x.val:  # x may already be moved due to overlapping
            #     x.val = x.val.move(self.act_home)
            # TODO: why if x.val ???
            x.val = x.val.move(self.act_home)
    
    def compute_layer(self,
                      token_idx:int,
                      layer_idx:int,
                      sub_batch_idx:int):
        self.layers[layer_idx].forward(self.hidden[token_idx][layer_idx][sub_batch_idx],
                                       self.cache_read_buf[layer_idx][sub_batch_idx],
                                       self.weight_read_buf[layer_idx],
                                       self.attention_mask[sub_batch_idx],
                                       self.cache_write_buf[layer_idx][sub_batch_idx],
                                       token_idx,
                                       sub_batch_idx)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def update_attention_mask(self, token_idx, sub_batch_idx):
        if token_idx > 0:
            mask = self.attention_mask[sub_batch_idx]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = sub_batch_idx * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = self.env.gpu
        val = attention_compute.allocate((self.policy.gpu_batch_size, self.task.prompt_len),
                                         torch.bool)
        val.load_from_torch((input_ids != self.config.pad_token_id))
        self.attention_mask[sub_batch_idx].store(val)

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            # print(f"DEBUG: Generation step {i}/{self.execute_gen_len-1}")

            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and torch.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)
    
    def generate(self,
                 inputs: torch.Tensor,
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: int = None,
                 cut_gen_len: int = None,
                 logits: bool = False,
                 verbose: int = 0):
        
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            logits=logits)
        
        self.set_task(task)

        tmp_batch_size = None
        if self.policy.gpu_batch_size * self.num_gpu_batches != len(task.inputs):
            tmp_batch_size = self.policy.gpu_batch_size
            self.policy.gpu_batch_size = len(task.inputs)
            
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)
        
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        pad_token_id = self.config.pad_token_id
        
        # WARNING: this is not allocated by our NUMA allocater
        # its located on default CPU device    
        # output_ids, stopped, 
        self.output_ids = torch.full((len(task.inputs), prompt_len + gen_len),
                                     pad_token_id,
                                     dtype=task.inputs[0].dtype)

        self.output_ids[:, :prompt_len] = task.inputs.to(self.output_ids.dtype)
        self.stopped = torch.zeros((len(task.inputs), 1), dtype=torch.bool)
        
        if logits:
            self.logits = torch.zeros((len(task.inputs),prompt_len, self.config.vocab_size),
                                      dtype=str_to_dtype[self.config.torch_dtype])
        else:
            self.logits = None

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
            
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
            
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
            
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)
       
        # Generate
        if not overlap:
            # No overlap, easy to understand, suitable for debugging
            self.generation_loop_normal()
        else:
            # Overlap I/O and compute
            if num_gpu_batches == 1:
                self.generation_loop_overlap_single_batch()
            else:
                self.generation_loop_overlap_multi_batch()

        # Delete cache
        for layer_idx in range(num_layers):
            for sub_batch_idx in range(num_gpu_batches):
                self.delete_cache(layer_idx, sub_batch_idx)

        if tmp_batch_size is not None:
            self.policy.gpu_batch_size = tmp_batch_size

        return self.output_ids, self.logits
    
    def __del__(self):
        self.delete_all_weights()