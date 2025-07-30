import argparse
import os, sys
import logging
from pathlib import Path
import json
from typing import Union, Optional
from functools import partial
from flexllmgen.flex_opt import add_parser_arguments

# dir_name = Path(__file__).parent.resolve()
# if not str(dir_name) in sys.path:
#     print(f"{dir_name} added to sys.path.")
#     sys.path.append(str(dir_name))
# from utils.model.flexllmgen1 import flexllmgen
import eval_utils.utils
from evaluator import (Evaluate, request_caching_arg_to_dict)
from eval_utils.tasks import (TaskManager, )
import eval_utils.api.registry
import eval_utils.api.model
from eval_utils.caching.cache import delete_cache
from eval_utils.loggers.evaluation_tracker import EvaluationTracker 
from eval_utils.loggers.wandb_logger import WandbLogger
from eval_utils.utils import (
    handle_non_serializable, 
    make_table,
    simple_parse_args_string, 
    convert_namespace_to_dict, 
)


eval_utils.utils.setup_logging()
eval_logger = logging.getLogger(__name__)

def try_parse_json(value: str) -> Union[str, dict, None]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise argparse.ArgumentTypeError(
                f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings."
            )
        return value
    
def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. "
            "Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(
            default_items[num_items:]
        )  # extend items list with missing defaults

    return items


def add_evalaute_parser_arguments():
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    
    parser.add_argument(
        "--model_type",
        "-mt",
        default="flexllmgen",
        type=str,
        help="Determine which model type to use, e.g. `hf`",
    )

    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="The task path to evaluate.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
    )

    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )

    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )

    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="Path where result metrics will be saved. Can be either a directory or a .json file. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--samples",
        "-E",
        default=None,
        type=str,
        metavar="/path/to/json",
        help='JSON string or path to JSON file containing doc indices of selected examples to test. Format: {"task_name":[indices],...}',
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--wandb_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--device", 
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--wandb_config_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to wandb.config.update. Use this to trace parameters that aren't already traced by default. eg. `lr=0.01,repeats=3",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=try_parse_json,
        default=None,
        help=(
            "Either comma delimited string or JSON formatted arguments for model generation on greedy_until tasks,"
            """ e.g. '{"temperature":0.7,"until":["hello"]}' or temperature=0,top_p=0.1."""
        ),
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=json.loads,
        default=None,
        help="""JSON string metadata to pass to task configs, for example '{"max_seq_lengths":[4096,8192]}'. Will be merged with model_args. Can also be set in task config.""",
    )
    parser.add_argument(
        "--log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )

    return parser


def get_model(
        args: argparse.Namespace,
        model_args: Optional[str] = None,
):
    assert isinstance(args.model_type, str), "model_type must be a string"
    if model_args is None:
            eval_logger.warning("model_args not specified. Using defaults.")
            model_args = ""
    else:
        assert args.device in [None, "cpu"] or "cuda" in args.device, f"device is {args.device},  should be assigned in  [None, 'cpu', 'cuda/cuda:N']"
        eval_logger.info(
            f"Initializing {args.model_type} model, with arguments: {simple_parse_args_string(model_args)}"
        )
        lm = eval_utils.api.registry.get_model(args.model_type).create_from_arg_string(
            model_args,
            {
                "batch_size": args.gpu_batch_size,
                "max_batch_size": args.gpu_batch_size,
                "device": args.device,
            },
        )
    # if args.model_type in ["flexllmgen", "flexllmgen_gpu", "flexllmgen-gpu", "flexllmgen_cxl", "flexllmgen-cxl"]:
    #     eval_logger.info(
    #             f"Initializing {args.model_type} model, with arguments: {simple_parse_args_string(model_args)}"
    #         )
    #     args.device = "flexllmgen use device GPU CPU and Disk, so set cuda"
    #     lm = eval_utils.api.registry.get_model(args.model_type).create_from_arg_string(
    #         model_args,
    #         {"args": args}
    #     )
    # else:
    #     if model_args is None:
    #         eval_logger.warning("model_args not specified. Using defaults.")
    #         model_args = ""
    #     else:
    #         assert args.device in [None, "cpu", "cuda"], f"device is {args.device},  should be assigned in  [None, 'cpu', 'cuda']"
    #         eval_logger.info(
    #             f"Initializing {args.model_type} model, with arguments: {simple_parse_args_string(model_args)}"
    #         )
    #         lm = eval_utils.api.registry.get_model(args.model_type).create_from_arg_string(
    #             model_args,
    #             {
    #                 "batch_size": args.gpu_batch_size,
    #                 "max_batch_size": args.gpu_batch_size,
    #                 "device": args.device,
    #             },
    #         )
    if args.use_cache is not None:
        eval_logger.info(f"Using cache at {args.use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = eval_utils.api.model.CachingLM(
            lm,
            args.use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )
    return lm



if __name__ == '__main__':

    parser = add_evalaute_parser_arguments()
    args = parser.parse_args()
    print(args)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()

    # 设置保存日志相关内容, 看是否启用wandb, 以及对保存结果的EvaluationTrackerc初始化
    if args.wandb_args:
        wandb_args_dict = simple_parse_args_string(args.wandb_args)
        wandb_config_args_dict = simple_parse_args_string(args.wandb_config_args)
        wandb_logger = WandbLogger(wandb_args_dict, wandb_config_args_dict)

    if args.output_path:
        args.log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(args.log_args)
    # eval_logger.info(f"{evaluation_tracker_args}, \n =============")
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)
    # exit()


    # 设置eval时的参数警告
    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.samples:
        assert args.limit is None, (
            "If --samples is not None, then --limit must be None."
        )
        if (samples := Path(args.samples)).is_file():
            args.samples = json.loads(samples.read_text())
        else:
            args.samples = json.loads(args.samples)

    # 设置TaskManager任务管理器
    metadata = (
        simple_parse_args_string(args.path)
    ) | (
        args.metadata
        if isinstance(args.metadata, dict)
        else simple_parse_args_string(args.metadata)
    )
    task_manager = TaskManager(include_path=args.include_path, metadata=metadata, data_path=args.data_path)
    if args.tasks == "list":
        print(task_manager.list_all_tasks())
        sys.exit()
    elif args.tasks == "list_groups":
        print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = eval_utils.utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            if task_names == ['xsum'] or task_names == 'xsum':
                eval_logger.warning(f"When Using unitxt==1.26.0, Better to install datasets==3.6.0 to avoid error: ImportError: cannot import name 'get_imports' from 'datasets.utils.py_utils'")
            # print(task_names)
            # print('=$'*20)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = eval_utils.utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{eval_utils.utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    # 定义是否需要重写request或删除request缓存    
    request_caching_args = request_caching_arg_to_dict(
        cache_requests=args.cache_requests
    )
    # model = flexllmgen(args)
    model_args = f"pretrained={args.path}"
    model = get_model(args, model_args)
    
    # final_params = {**convert_namespace_to_dict(args), **request_caching_args, "task_manager":task_manager, "evaluation_tracker":evaluation_tracker}

    evaluator = Evaluate(
        args=args,
        model=model,
        model_args=model_args,
        tasks=task_names,
        batch_sizes=args.gpu_batch_size * args.num_gpu_batches,
        use_cache=args.use_cache,
        limit=args.limit,
        samples=args.samples,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        metadata=metadata,
        **request_caching_args,
    )
    # exit()
    results = evaluator.process_all_outputs()
    results = evaluator.postprocess_results(results)
    

    # 有结果的话就输出结果
    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if args.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

        print(
            f"{args.model} ({model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.gpu_batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if args.wandb_args:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()
