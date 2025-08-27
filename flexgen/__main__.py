import argparse
import torch

from flexgen.models import get_model_architecture
from flexgen.models.load_convert.load_weight import AutoFlexModel
from flexgen.models.utils import Policy, ExecutionEnv, get_tokenizer
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, fix_recursive_import)
from flexgen.utils import (str2bool, project_decode_latency, write_benchmark_log, get_filename,  
                           DUMMY_WEIGHT, GB) 
from flexgen.models.config import FlexModelConfig
from flexgen.timer import timers


fix_recursive_import()

def add_parser_arguments(parser:argparse.ArgumentParser):
    parser.add_argument("--model", type=str, default="llama2-7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="/shared/model/Llama-2-7b-hf",
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
    parser.add_argument("--gpu-batch-size", type=int, default=2)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 0, 0, 0, 100, 100, 0, 0],
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
    # prompts = ["Paris is the capital city of"]
    # prompts = ["Summarize the following document: Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday.\nDetectives said three firearms, ammunition and a five-figure sum of money were recovered.\nA 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday..\n"]
    # prompts.append(["Summarize the following document: Prison Link Cymru had 1,099 referrals in 2015-16 and said some ex-offenders were living rough for up to a year before finding suitable accommodation.\nWorkers at the charity claim investment in housing would be cheaper than jailing homeless repeat offenders.\nThe Welsh Government said more people than ever were getting help to address housing problems.\nChanges to the Housing Act in Wales, introduced in 2015, removed the right for prison leavers to be given priority for accommodation.\nPrison Link Cymru, which helps people find accommodation after their release, said things were generally good for women because issues such as children or domestic violence were now considered.\nHowever, the same could not be said for men, the charity said, because issues which often affect them, such as post traumatic stress disorder or drug dependency, were often viewed as less of a priority.\nAndrew Stevens, who works in Welsh prisons trying to secure housing for prison leavers, said the need for accommodation was \"chronic\".\n\"There's a desperate need for it, finding suitable accommodation for those leaving prison there is just a lack of it everywhere,\" he said.\n\"It could take six months to a year, without a lot of help they could be on the streets for six months.\n\"When you think of the consequences of either being on the street, especially with the cold weather at the moment or you may have a roof over your head, sometimes there is only one choice.\"\nMr Stevens believes building more one-bedroom flats could help ease the problem.\n\"The average price is a hundred pounds a week to keep someone in a rented flat, prison is a lot more than that so I would imagine it would save the public purse quite a few pounds,\" he said.\nOfficial figures show 830 one-bedroom properties were built in the year to March 2016, of an overall total of 6,900 new properties in Wales.\nMarc, 50, who has been in and out of prison for the past 20 years for burglary offences, said he struggled to find accommodation each time he was released.\nHe said he would ask himself: \"Where am I going to stay? Where am I going to live? Have I got somewhere where I can see my daughter.\"\n\"You're put out among the same sort of people doing the same sort of thing, and it's difficult, it's difficult to get away from it. It's like every man for himself, there's nothing.\"\nMarc has now found stable accommodation with homeless charity Emmaus and said it had been life changing.\n\"You feel safe, you got hot food, you've got company of people in similar situations to yourself but all dealing with different issues. It's a constructive, helpful atmosphere,\" he said.\nTom Clarke, chief executive of Emmaus South Wales, agreed there was not enough support available.\n\"We do still see [people] homeless on the streets, so clearly they haven't got accommodation and haven't got provision,\" he said.\n\"I think the key is connecting people with the services they need. I don't delude myself that Emmaus can offer a one size fits all for everyone, we can't.\n\"But there must be other opportunities and given suitable encouragement I believe that can and should happen.\"\nA Welsh Government spokesman said the national pathway for homeless services to children, young people and adults in the secure estate had prevented many people from losing their home whilst serving their prison sentence.\nIt added there were already significant demands for one-bedroom flats across the public and private sector and it was providing 20,000 new affordable homes in the next five years..\n"])
    prompt1 = ("Summarize the following document: Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday.\nDetectives said three firearms, ammunition and a five-figure sum of money were recovered.\nA 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday..\n")
    prompt2 = ("Summarize the following document: Prison Link Cymru had 1,099 referrals in 2015-16 and said some ex-offenders were living rough for up to a year before finding suitable accommodation.\nWorkers at the charity claim investment in housing would be cheaper than jailing homeless repeat offenders.\nThe Welsh Government said more people than ever were getting help to address housing problems.\nChanges to the Housing Act in Wales, introduced in 2015, removed the right for prison leavers to be given priority for accommodation.\nPrison Link Cymru, which helps people find accommodation after their release, said things were generally good for women because issues such as children or domestic violence were now considered.\nHowever, the same could not be said for men, the charity said, because issues which often affect them, such as post traumatic stress disorder or drug dependency, were often viewed as less of a priority.\nAndrew Stevens, who works in Welsh prisons trying to secure housing for prison leavers, said the need for accommodation was \"chronic\".\n\"There's a desperate need for it, finding suitable accommodation for those leaving prison there is just a lack of it everywhere,\" he said.\n\"It could take six months to a year, without a lot of help they could be on the streets for six months.\n\"When you think of the consequences of either being on the street, especially with the cold weather at the moment or you may have a roof over your head, sometimes there is only one choice.\"\nMr Stevens believes building more one-bedroom flats could help ease the problem.\n\"The average price is a hundred pounds a week to keep someone in a rented flat, prison is a lot more than that so I would imagine it would save the public purse quite a few pounds,\" he said.\nOfficial figures show 830 one-bedroom properties were built in the year to March 2016, of an overall total of 6,900 new properties in Wales.\nMarc, 50, who has been in and out of prison for the past 20 years for burglary offences, said he struggled to find accommodation each time he was released.\nHe said he would ask himself: \"Where am I going to stay? Where am I going to live? Have I got somewhere where I can see my daughter.\"\n\"You're put out among the same sort of people doing the same sort of thing, and it's difficult, it's difficult to get away from it. It's like every man for himself, there's nothing.\"\nMarc has now found stable accommodation with homeless charity Emmaus and said it had been life changing.\n\"You feel safe, you got hot food, you've got company of people in similar situations to yourself but all dealing with different issues. It's a constructive, helpful atmosphere,\" he said.\nTom Clarke, chief executive of Emmaus South Wales, agreed there was not enough support available.\n\"We do still see [people] homeless on the streets, so clearly they haven't got accommodation and haven't got provision,\" he said.\n\"I think the key is connecting people with the services they need. I don't delude myself that Emmaus can offer a one size fits all for everyone, we can't.\n\"But there must be other opportunities and given suitable encouragement I believe that can and should happen.\"\nA Welsh Government spokesman said the national pathway for homeless services to children, young people and adults in the secure estate had prevented many people from losing their home whilst serving their prison sentence.\nIt added there were already significant demands for one-bedroom flats across the public and private sector and it was providing 20,000 new affordable homes in the next five years..\n")
    prompts = [prompt1, prompt2]
    
    # 使用tokenizer处理两个prompt
    input_ids = tokenizer(prompts, 
                          max_length=prompt_len,
                          padding=True,
                          truncation=True,
                          return_tensors="pt")
    
    # # 打印每个prompt的tokenized结果
    # print("=== Tokenization Results ===")
    # for i, prompt in enumerate(prompts):
    #     print(f"Prompt {i+1}: {prompt[:100]}...")
    #     print(f"Tokenized {i+1}: {input_ids.input_ids[i]}")
    #     print(f"Length {i+1}: {len(input_ids.input_ids[i])}")
    #     print("---")
    
    # 返回两个prompt的tokenized结果，而不是重复第一个
    return tuple(input_ids.input_ids)


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    # if args.model == "facebook/galactica-30b":
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    # tokenizer = AutoTokenizer.from_pretrained(args.path, padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    

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

    print("init weight...")
    model_config, converted_path, weight_map = AutoFlexModel.from_pretrained(args.path, force_convert=args.force_convert, save_prefix=args.save_prefix)
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

    tokenizer = get_tokenizer(args.path)
    # tokenizer = AutoTokenizer.from_pretrained(args.path)
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    # print('=========================')
    # print(f"inputs: {inputs}")
    # print(f"tranformers inputs is: [ 1, 3681, 338, 278, 7483, 4272, 310]")
    # is_equal = inputs == ([ 1, 3681, 338, 278, 7483, 4272, 310], )
    # print(f"inputs and transformers inputs id is_equal: {is_equal}")
    # print('=========================')

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
        # print(f"output_ids: {output_ids}")
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
    cxl.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        model_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(20)
    
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 9

    run_flexgen(args)
