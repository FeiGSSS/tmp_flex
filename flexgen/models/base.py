from pathlib import Path
import numpy as np
import dataclasses
from typing import Any, List, Union, Optional
import torch
from tqdm import tqdm

from flexgen.utils import (ValueHolder, 
                           array_1d, array_2d, array_3d, array_4d
                           )
# from flexgen.compression import CompressionConfig
from flexgen.timer import timers
from flexgen.models.config import FlexModelConfig
from flexgen.models.utils import ExecutionEnv, Policy, Task
# from flexgen.flex_model import Policy

# 假设这些依赖项在您的主项目中可用
# from flex_model_config import FlexModelConfig
# from flex_model import Policy, init_weight_list

# timers = Timers()


class BaseModelLayer:
    """所有模型层的基类，共享通用属性。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict):
        self.config = config
        self.env = env
        self.policy = policy
        self.weight_map = weight_map
        self.compute_device = self.env.gpu
        self.weight_load_dst = (self.compute_device.compressed_device if policy.compress_weight
            else self.compute_device)
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, converted_path: str):
        """
        为该层初始化权重。
        每个子类都必须实现这个方法来定义自己的权重规格。
        """
        raise NotImplementedError("Each model layer must implement its own init_weight method.")
    def load_weight(self, 
                    weight_home:ValueHolder, 
                    weight_read_buf:ValueHolder, 
                    k:List[Union[int, float]]):
        pass  # do nothing
    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        pass  # do nothing

    def forward(self, ):
        pass  # do nothing


class BaseTransformerLayer(BaseModelLayer):
    """Transformer层的基类。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map: dict):
        super().__init__(config, env, policy, weight_map=weight_map)
        # 子类将负责实例化具体的 attention 和 mlp 层
        self.attention = None
        self.mlp = None

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)
    def init_weight(self, weight_home, converted_path: str):
        """组合 attention 和 mlp 层的权重初始化。"""
        # ValueHolder 是 FlexGen 中用于管理张量的辅助类
        """
        为该层初始化权重。
        每个子类都必须实现这个方法来定义自己的权重规格。
        """
        raise NotImplementedError("Each model layer must implement its own init_weight method.")

    def forward(self, ):
        pass  # do nothing

class BaseModel:
    """所有模型的基类，共享通用属性。"""
    def __init__(self, 
                 config:FlexModelConfig, 
                 env:ExecutionEnv, 
                 policy:Policy, 
                 weight_map:dict, 
                 path:str):
        self.config = config
        self.env = env
        self.compute_device = self.env.gpu
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.weight_map = weight_map
        self.path = path
        self.layers:List[Union[BaseModelLayer, BaseTransformerLayer]] = []
        self.task = None

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

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)
    
    @property
    def num_layers(self, ):
        return len(self.layers)
    
    def init_cache_area(self, ):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        # cache[j][k]
        self.cache_home = array_2d(self.num_layers, self.num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(self.num_layers, self.num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(self.num_layers, self.num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(self.num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(self.num_gpu_batches, ValueHolder)


    def get_hidden_area(self, cls, i):
        return cls(*i, ValueHolder)
    
    def load_weight(self,  
                    i:List[Union[int, float]], 
                    j:List[Union[int, float]], 
                    k:List[Union[int, float]], 
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
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        try:
            if k == 0:
                for x in self.weight_home[j].pop():
                    if isinstance(x, ValueHolder):
                        for y in x.pop():
                            y.delete()
                    else:
                        x.delete()
        except TypeError:
        # 这个 TypeError 只会在解释器关闭、模块已被卸载时发生。
        # 此时程序即将退出，内存将由操作系统回收，
        # 所以我们可以安全地忽略这个错误，以避免程序崩溃。
            
            pass
    
    def init_cache(self, 
                   j:List[Union[int, float]], 
                   k:List[Union[int, float]]):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self,  
                   i:List[Union[int, float]], 
                   j:List[Union[int, float]], 
                   k:List[Union[int, float]], 
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
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self,  
                    i:List[Union[int, float]], 
                    j:List[Union[int, float]], 
                    k:List[Union[int, float]], 
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
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, 
                   i:List[Union[int, float]], 
                   j:List[Union[int, float]], 
                   k:List[Union[int, float]], ):
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
        dst = self.layers[j].compute_device
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

            if logits is not None:
                self.logits[left:right] = logits

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
    
    # def load_weight(self, i, j, k, overlap=True):
    #     raise NotImplementedError("Func load_weight must be implemented by subclasses")
    
    # def delete_weight(self, j, k):
    #     raise NotImplementedError("Func delete_weight must be implemented by subclasses")

    # def init_cache(self, j, k):
    #     raise NotImplementedError("Func init_cache must be implemented by subclasses")

    # def load_cache(self, i, j, k, overlap=True):
    #     raise NotImplementedError("Func load_cache must be implemented by subclasses")

    # def store_cache(self, i, j, k, overlap=True):
    #     raise NotImplementedError("Func store_cache must be implemented by subclasses")

    # def delete_cache(self, j, k):
    #    raise NotImplementedError("Func delete_cache must be implemented by subclasses")


    # def load_hidden(self, i, j, k):
    #     raise NotImplementedError("Func load_hidden must be implemented by subclasses")


    # def store_hidden(self, i, j, k):
    #     raise NotImplementedError("Func store_hidden must be implemented by subclasses")


    # def compute_layer(self, i, j, k):
    #     raise NotImplementedError("Func compute_layer must be implemented by subclasses")

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

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
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

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)
    
    def __del__(self):
        self.delete_all_weights()

    def get_logits(self, inputs: Union[np.array, List[List[int]]]):
        
        max_new_tokens = 1
        do_sample = False
        temperature = 1.0
        stop = None
        cut_gen_len = 1
        
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            logits=True
        )
        
        tmp_batch_size = None
        if self.policy.gpu_batch_size * self.num_gpu_batches != len(task.inputs):
            tmp_batch_size = self.policy.gpu_batch_size
            self.policy.gpu_batch_size = len(task.inputs)
        num_layers = self.num_layers
        num_batches = self.num_gpu_batches
        batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        
        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
                                  self.config.pad_token_id,
                                  dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        self.logits = np.zeros((len(task.inputs), prompt_len, self.config.vocab_size), dtype=np.float32)
        # print(f"self.logits shape: {self.logits.shape}, {prompt_len}, {gen_len} \n =========***********");exit()
        
        # if len(task.inputs) != batch_size:
        #     batch_size = len(task.inputs)
            # self.policy.batch_size = batch_size
        assert batch_size * num_batches == len(task.inputs), f"batch_size * num_batches != len(task.inputs)! batch_size:{batch_size}, num_batches:{num_batches}, len(task.inputs):{len(task.inputs)}"
        
        for j in range(num_layers):
            for k in range(num_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
                
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
            
        for k in range(num_batches):
            self.attention_mask[k].clear()
            
        self.hidden = array_3d(gen_len, num_layers, num_batches, ValueHolder)
        
        self.set_task(task)
        
        for j in range(num_layers):
            for k in range(num_batches):
                self.init_cache(j, k)
        
        self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)
        
        if not overlap:
            # No overlap, easy to understand, suitable for debugging
            self.generation_loop_normal()
        else:
            # Overlap I/O and compute
            if num_batches == 1:
                self.generation_loop_overlap_single_batch()
            else:
                self.generation_loop_overlap_multi_batch()
                # raise NotImplementedError("Only support num_batches=1 for now")
                # self.generation_loop_overlap_multi_batch()
        
        # Delete cache
        for j in range(num_layers):
            for k in range(num_batches):
                self.delete_cache(j, k)
                
        self.env.cpu.del_attention_compute_workspace()
        if tmp_batch_size is not None:
            self.policy.gpu_batch_size = tmp_batch_size

        return self.output_ids, self.logits
    
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
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
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

        return self.output_ids
    

    


