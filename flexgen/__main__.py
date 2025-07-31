import argparse
import dataclasses
import os
import pickle as pkl
import time
from typing import Union, List, Optional
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.models import get_model_architecture
from flexgen.models.load_convert.load_weight import AutoFlexModel
from flexgen.models.utils import Policy, ExecutionEnv
from flexgen.pytorch_backend import (TorchDevice, TorchTensor, TorchDisk, TorchNuma, TorchMixedDevice, 
                                     fix_recursive_import, print_memory_copy_stats)
from flexgen.utils import (str2bool, project_decode_latency, write_benchmark_log, get_filename,  
                           DUMMY_WEIGHT, GB) 
from flexgen.models.config import FlexModelConfig
from flexgen.compression import CompressionConfig
from flexgen.timer import timers


fix_recursive_import()

def add_parser_arguments(parser:argparse.ArgumentParser):
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
        help="The model name.")
    parser.add_argument("--path", type=str, default="/shared/model/opt/opt-125m",
        help="The path to the model weights. If there are no cached weights, "
             "flexgen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=26)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 0, 100, 0, 0, 100, 0, 0],
        help="Nine numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of weight on NUMA, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of attention cache on NUMA, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU, "
         "the percentage of activations on NUMA")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)




def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    # tokenizer = AutoTokenizer.from_pretrained(args.path, padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    numa = TorchNuma()
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, numa=numa, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1], args.percent[2],
                    args.percent[3], args.percent[4], args.percent[5],
                    args.percent[6], args.percent[7], args.percent[8],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    print("init weight...")
    model_config, converted_path, weight_map = AutoFlexModel.from_pretrained(args.path, force_convert=True)
    model_config: FlexModelConfig
    converted_path: str
    weight_map: dict

    # opt_config = get_opt_config(args.model)
    # cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    # hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    cache_size = model_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = model_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {model_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("load weight...")
    model = get_model_architecture(config=model_config, path=converted_path, policy=policy, env=env, weight_map=weight_map)
    # model = OptLM(opt_config, env, args.path, policy)
    # print(model_config.__dict__)
    # exit()
    try:
        # print("warmup - generate")
        # output_ids = model.generate(
        #     warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        model_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)
    
    print_memory_copy_stats()


if __name__ == "__main__":
    import torch
    torch.set_num_threads(20)
    
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 9

    run_flexgen(args)
