import argparse
import torch

from flexgen.models import get_model_architecture
from flexgen.models.utils import Policy, ExecutionEnv
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, fix_recursive_import)
from flexgen.utils import (str2bool, project_decode_latency, write_benchmark_log, get_filename, DUMMY_WEIGHT, GB) 
from flexgen.timer import timers


fix_recursive_import()

def add_parser_arguments(parser:argparse.ArgumentParser):
    parser.add_argument("--path", type=str, default="/shared/model/Llama-2-7b-hf",
                        help="The path to the model weights. If there are no cached weights, "
                        "flexgen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="./flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[50, 50, 0, 0, 0, 100, 100, 0, 0],
        help="Nine numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of weight on CXL, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of attention cache on CXL, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU, "
         "the percentage of activations on CXL")
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
    parser.add_argument("--force_convert", default=False, action="store_true",
                        help="Whether to force convert the model weights.")
    parser.add_argument("--save_prefix", default=None, type=str,
                        help="Whether to force convert the model weights.")




def get_test_inputs(prompt_len, num_prompts, tokenizer):

    prompts = [
        "问题: 一杯冷水放在室温的房间里几个小时后，水温会趋近于什么？\n回答:",
        "问题: 为什么天空在晴朗白天呈现蓝色？一句话解释。\n回答:",
        "问题: 设一个等差数列首项为 3，公差为 5，第 20 项是多少？\n回答:", #"3 + 19*5 = 98"
        "问题: 今天是星期三，10 天后是星期几？请给出推理步骤。\n回答:",  # "10 ≡ 3 (mod 7)，星期三往后 3 天是星期六"
    ]

    prompts = prompts[:num_prompts]
    
    input_ids = tokenizer(prompts, 
                          max_length=prompt_len,
                          padding=True,
                          truncation=True,
                          return_tensors="pt")

    return input_ids.input_ids


def main(args):
    
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    cxl = TorchDevice("cxl")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, cxl=cxl)

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1], args.percent[2],
                    args.percent[3], args.percent[4], args.percent[5],
                    args.percent[6], args.percent[7], args.percent[8],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity)
    
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    print("init model...")
    model = get_model_architecture(pretrained_model_path=args.path, env=env, policy=policy)
    
    cache_size = model.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = model.hidden_bytes(num_prompts, prompt_len + gen_len)
    model_bytes = model.model_bytes()
    print(f"model size: {model_bytes/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    tokenizer = model.tokenizer
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    try:
        print("benchmark - generate")
        timers("generate").reset()
        output_ids, logits = model.generate(inputs=inputs,
                                            max_new_tokens=args.gen_len,
                                            cut_gen_len=cut_gen_len,
                                            stop=model.config.eos_token_id)
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

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in range(len(outputs)):
            show_str += f"{i}:\n {outputs[i]}\n"
            show_str += "=*" * 20 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    cxl.print_stats()


if __name__ == "__main__":
    import torch
    torch.set_num_threads(20)
    
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 9

    main(args)
