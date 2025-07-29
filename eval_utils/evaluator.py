import itertools
import json
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch

# import sys
# from pathlib import Path
# dir_name = Path(__file__).parent.resolve()
# if not str(dir_name) in sys.path:
#     print(f"{dir_name} added to sys.path.")
#     sys.path.append(str(dir_name))
# print(dir_name)
# print(sys.path)
import eval_utils.utils
import eval_utils.api.metrics
import eval_utils.api.registry
import eval_utils.api.task
import eval_utils.model
from eval_utils.caching.cache import delete_cache
from evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from eval_utils.loggers.evaluation_tracker import EvaluationTracker
from eval_utils.loggers import add_env_info, add_tokenizer_info, get_git_commit_hash
from eval_utils.tasks import TaskManager, get_task_dict
from eval_utils.utils import (
    handle_non_serializable,
    hash_string,
    positional_deprecated,
    setup_logging,
    simple_parse_args_string,
)


if TYPE_CHECKING:
    from eval_utils.api.model import LM
    from eval_utils.api.task import Task

setup_logging()
eval_logger = logging.getLogger(__name__)

class Simple_Evaluate():
    def __init__(self, 
        args, 
        model, 
        model_args: Optional[Union[str, dict]] = None,
        tasks: Optional[List[Union[str, dict, object]]] = None,
        use_cache: Optional[str] = None,
        batch_sizes: Optional[Union[int, List[int]]] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        num_fewshot: Optional[int] = None,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        gen_kwargs: Union[str, dict, None] = None,
        samples: Optional[dict] = None,
        bootstrap_iters: int = 100000,
        check_integrity: bool = False,
        write_out: bool = False,
        log_samples: bool = True,
        evaluation_tracker: Optional[EvaluationTracker] = None,
        system_instruction: Optional[str] = None,
        apply_chat_template: Union[bool, str] = False,
        fewshot_as_multiturn: bool = False,
        task_manager: Optional[TaskManager] = None,
        verbosity=None,
        predict_only: bool = False,
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        fewshot_random_seed: int = 1234,
        confirm_run_unsafe_code: bool = False,
        metadata: Optional[dict] = None,
        **kwargs,
                 ):
        self.args = args
        self.model = model
        self.model_args = model_args
        self.tasks = tasks
        self.use_cache = use_cache
        self.batch_sizes = batch_sizes
        self.cache_requests = cache_requests
        self.rewrite_requests_cache = rewrite_requests_cache
        self.delete_requests_cache = delete_requests_cache
        self.limit = limit
        self.num_fewshot = num_fewshot
        self.gen_kwargs = gen_kwargs
        self.samples = samples
        self.bootstrap_iters = bootstrap_iters
        self.check_integrity = check_integrity
        self.write_out = write_out
        self.log_samples = True if predict_only else log_samples
        self.predict_only = predict_only
        self.evaluation_tracker = evaluation_tracker
        self.system_instruction = system_instruction
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.task_manager = task_manager
        self.verbosity = verbosity
        self.random_seed = random_seed
        self.numpy_random_seed = numpy_random_seed
        self.torch_random_seed = torch_random_seed
        self.fewshot_random_seed = fewshot_random_seed
        self.confirm_run_unsafe_code = confirm_run_unsafe_code
        self.metadata = metadata
        self.verify_task_and_taskmanager()
        self.random_seeds = self.set_random_seeds()
        
        task_dict = get_task_dict(self.tasks, self.task_manager)
        self.task_dict = self._adjust_config(task_dict)
        
        self.evaluation_tracker_log_args()

    def _adjust_config(self, 
        task_dict, 
        ):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: self._adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if self.gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=self.gen_kwargs, update=True
                        )
                    eval_logger.info(
                        f"{task_obj.config.task}: Using gen_kwargs: {task_obj.config.generation_kwargs}"
                    )

                if self.predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if self.num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        eval_logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {self.num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=self.num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=self.fewshot_random_seed)

                adjusted_task_dict[task_name] = task_obj
        eval_logger.info(f"Adjusted task configs done")
        return adjusted_task_dict


    
    def verify_task_and_taskmanager(self, ):
        if self.tasks is None:
            self.tasks = []
        if len(self.tasks) == 0:
            raise ValueError(
                "No tasks specified, or no tasks found. Please verify the task names."
            )
        eval_logger.info(f"task is not None, the selected task is {self.tasks}")
        if self.task_manager is None:
            raise ValueError(
                "No task_manager specified. Please construct TaskManager."
            )
        eval_logger.info(f"Successfully Verified task and task_manager")

    def _delete_cache(self, ):
        if self.delete_requests_cache:
            delete_cache()
    
    def set_random_seeds(self, ):
        seed_message = []
        if self.random_seed is not None:
            # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
            seed_message.append(f"Setting random seed to {self.random_seed}")
            random.seed(self.random_seed)

        if self.numpy_random_seed is not None:
            seed_message.append(f"Setting numpy seed to {self.numpy_random_seed}")
            np.random.seed(self.numpy_random_seed)

        if self.torch_random_seed is not None:
            seed_message.append(f"Setting torch manual seed to {self.torch_random_seed}")
            torch.manual_seed(self.torch_random_seed)

        if self.fewshot_random_seed is not None:
            seed_message.append(f"Setting fewshot manual seed to {self.fewshot_random_seed}")
        return seed_message

    def evaluation_tracker_log_args(self, ):
        if self.evaluation_tracker is not None:
            # print(self.model_args)
            # print('=*'*20)
            self.evaluation_tracker.general_config_tracker.log_experiment_args(
                model_source=self.args.model_type,
                model_args=self.model_args,
                system_instruction=self.system_instruction,
                chat_template=self.model.chat_template(self.apply_chat_template)
                if self.apply_chat_template
                else None,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
            )



class Evaluate(Simple_Evaluate):
    def __init__(self,
        *args, **kwargs,
    ):
        # 将所有参数传递给父类 Simple_Evaluate 的 __init__ 方法
        super().__init__(
            *args, **kwargs,
        )

        self.start_date = time.time()
        # tracks all Instances/requests a model must generate output on.
        self.requests = defaultdict(list)
        # stores the amount to pad out reqs per req. type so that
        # number of fwd passes per distributed rank is equal
        self.padding_requests = defaultdict(int)
        self.latancy_throughputs = defaultdict()
        
        # get lists of group hierarchy and each type of request
        self.eval_tasks = get_task_list(self.task_dict)
        if not self.log_samples:
            if not all(
                "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
                for task_output in self.eval_tasks
            ):
                raise ValueError("log_samples must be True for 'bypass' metric-only tasks")
        
        # validation checks:
        # 1.are we running multimodal task <-> non-multimodal model class, or vice-versa.
        # 2.are we running code that is marked as unsafe.
        self.validation_checks()

        # Cache the limit arg.
        self.limits = self.cache_limit_args()

    def validation_checks(self, ):
        incompatible_tasks = []
        for task_output in self.eval_tasks:
            task: Task = task_output.task

            if getattr(task, "MULTIMODAL", False) and not getattr(self.model, "MULTIMODAL", False):
                incompatible_tasks.append(task_output.task_name)
        if len(incompatible_tasks) > 0:
            if not getattr(self.model, "MULTIMODAL", False):
                raise ValueError(
                    f"Attempted to run tasks: {incompatible_tasks} which require multimodal input, but the selected model type does not currently implement this. Multimodal support is currently restricted to the ['hf-multimodal', 'vllm-vlm'] model type."
                )
    
    def cache_limit_args(self, ):
        limit_arg = self.limit
        limits = []
        for task_output in self.eval_tasks:
            task: Task = task_output.task

            limit = get_sample_size(task, limit_arg)
            limits.append(limit)
            task.build_all_requests(
                limit=limit,
                samples=self.samples.get(task_output.task_name, None)
                if self.samples is not None
                else self.samples,
                rank=self.model.rank,
                world_size=self.model.world_size,
                cache_requests=self.cache_requests,
                rewrite_requests_cache=self.rewrite_requests_cache,
                system_instruction=self.system_instruction,
                apply_chat_template=bool(self.apply_chat_template),
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                chat_template=getattr(self.model, "apply_chat_template")
                if self.apply_chat_template
                else None,
                tokenizer_name=getattr(self.model, "tokenizer_name", "")
                if self.apply_chat_template
                else "",
            )
            eval_logger.debug(
                f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
            )
            if self.write_out:
                print_writeout(task)
            # aggregate Instances by LM method requested to get output.
            for instance in task.instances:
                reqtype = instance.request_type
                self.requests[reqtype].append(instance)
        return limits

    def get_all_outputs(self, ):

        # set model throughput and latancy collection list
        flag = 0
        latancy_dict = defaultdict(list)
        throughput_dict = defaultdict(list)


        # we need to run process_all_outputs() after get all outputs, before postprocessing results.
        for reqtype, reqs in self.requests.items():
            if flag == 1:
                latancy_dict = defaultdict(list)
                throughput_dict = defaultdict(list)
            flag = 1
            eval_logger.info(f"Running {reqtype} requests")
            # shortest_key = "s"*100
            # create `K` copies of each request `req` based off `K = req.repeats`
            cloned_reqs = []
            for req in reqs:
                # shortest_key = min(shortest_key, req.task_name, key=len)
                cloned_reqs.extend([req] * req.repeats)

            # run requests through model
            latancy_throughputs = None
            try:
                resps, latancy_throughputs = getattr(self.model, reqtype)(cloned_reqs)
            except:
                resps = getattr(self.model, reqtype)(cloned_reqs)
            if latancy_throughputs is not None:
                for batch_lt in latancy_throughputs:
                    latancy_dict['prefill'].append(batch_lt[0])
                    latancy_dict['decode'].append(batch_lt[1])
                    latancy_dict['total'].append(batch_lt[2])
                    throughput_dict['prefill'].append(batch_lt[3])
                    throughput_dict['decode'].append(batch_lt[4])
                    throughput_dict['total'].append(batch_lt[5])
                latancy_throughputs = self.compute_latency_throughput(latancy_dict, throughput_dict)
                
            # assert req.task_name not in self.latancy_throughputs.keys(), f"{req.task_name} has already been computed latancy_throughput, Please check"

            self.latancy_throughputs[reqtype] = latancy_throughputs
            # if len(resps)>2:
            #     resps, latancy_throughputs = resps
            # put responses from model into a list of length K for each request.
            for x, req in zip(resps, cloned_reqs):
                req.resps.append(x)
        # return resps, cloned_reqs

    def process_all_outputs(self, ):
        # process all outputs after getting all outputs, before postprocessing results.
        # resps, cloned_reqs = self.get_all_outputs()
        self.get_all_outputs()
        for task_output, limit in zip(self.eval_tasks, self.limits):
            task = task_output.task
            task.apply_filters()
            
            ### Collect values of metrics on all datapoints ###
            # # unpack results and sort back in order and return control to Task
            # TODO: make it possible to use a different metric per filter
            # Pre-process task.instances to group by doc_id
            instances_by_doc_id = defaultdict(list)
            for instance in task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            # Sort instances within each group
            for instances in instances_by_doc_id.values():
                instances.sort(key=lambda x: x.idx)
            # iterate over different filters used
            for filter_key in task.instances[0].filtered_resps.keys():
                indices = (
                    self.samples.get(task_output.task_name, None)
                    if self.samples is not None
                    else None
                )
                doc_iterator = task.doc_iterator(
                    rank=self.model.rank, 
                    limit=limit,
                    world_size=self.model.world_size,
                    samples=indices,

                )
                for doc_id, doc in doc_iterator:
                    if indices:
                        doc_id_true = indices[doc_id]
                    else:
                        doc_id_true = doc_id
                    requests = instances_by_doc_id[doc_id]
                    metrics = task.process_results(
                        doc, [req.filtered_resps[filter_key] for req in requests]
                    )
                    if self.log_samples:
                        target = task.doc_to_target(doc)
                        example = {
                            "doc_id": doc_id_true,
                            "doc": doc,
                            "target": target,
                            "arguments": [req.args for req in requests],
                            "resps": [req.resps for req in requests],
                            "filtered_resps": [
                                req.filtered_resps[filter_key] for req in requests
                            ],
                            "filter": filter_key,
                            "metrics": list(metrics.keys()),
                            "doc_hash": hash_string(
                                json.dumps(
                                    requests[0].doc,
                                    indent=2,
                                    default=handle_non_serializable,
                                    ensure_ascii=False,
                                )
                            ),
                            "prompt_hash": hash_string(requests[0].arguments[0]),
                            "target_hash": hash_string(str(target)),
                        }
                        example.update(metrics)
                        task_output.logged_samples.append(example)
                    for metric, value in metrics.items():
                        task_output.sample_metrics[(metric, filter_key)].append(value)
        if self.model.rank == 0:
            ### Aggregate results over all datapoints ###
            # aggregate results ; run bootstrap CIs
            for task_output in self.eval_tasks:
                task_output.calculate_aggregate_metric(bootstrap_iters=self.bootstrap_iters)
            (
                results,
                samples,
                configs,
                versions,
                num_fewshot,
                higher_is_better,
            ) = consolidate_results(self.eval_tasks)

            ### Calculate group metrics ###
            if bool(results):
                results, versions, show_group_table, *_ = consolidate_group_results(
                    results, versions, self.task_dict
                )

            results_agg, group_agg = prepare_print_tasks(self.task_dict, results)
            subtask_list = get_subtask_list(self.task_dict)

            # collect all higher_is_better values for metrics
            # in the group's subtasks.
            # TODO: clean this up ; unify with the below metric_list loop?
            _higher_is_better = {}
            for group, task_list in subtask_list.items():
                if (
                    len(task_list) != 0
                ):  # subtask list will list "task_name": [] for solo tasks
                    for task in task_list:
                        for m, h in higher_is_better[task].items():
                            if m not in _higher_is_better.keys():
                                _higher_is_better[m] = h

                            if (
                                m in _higher_is_better
                                and _higher_is_better[m] is not None
                                and _higher_is_better[m] != h
                            ):
                                eval_logger.warning(
                                    f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None."
                                )
                                _higher_is_better[m] = None
                    higher_is_better[group] = _higher_is_better
            
            
            for key in self.latancy_throughputs.keys():
                higher_is_better[key]['prefill_latency/TTFT_mean'] = False
                higher_is_better[key]['prefill_latency/TTFT_median'] = False
                higher_is_better[key]['prefill_latency/TTFT_p99'] = False
                higher_is_better[key]['prefill_throughput'] = True

                higher_is_better[key]['decode_latency/TPOT_mean'] = False
                higher_is_better[key]['decode_latency/TPOT_median'] = False
                higher_is_better[key]['decode_latency/TPOT_p99'] = False
                higher_is_better[key]['decode_throughput'] = True

                higher_is_better[key]['total_latency_mean'] = False
                higher_is_better[key]['total_latency_median'] = False
                higher_is_better[key]['total_latency'] = False
                higher_is_better[key]['total_throughput'] = True
                if bool(group_agg) & show_group_table:
                    if key not in group_agg.keys():
                        group_agg[key] = {}
                    group_agg[key]['prefill_latency/TTFT_mean'] = self.latancy_throughputs[key]['prefill_latency']['mean']
                    group_agg[key]['prefill_latency/TTFT_median'] = self.latancy_throughputs[key]['prefill_latency']['median']
                    group_agg[key]['prefill_latency/TTFT_p99'] = self.latancy_throughputs[key]['prefill_latency']['p99']
                    group_agg[key]['prefill_throughput'] = self.latancy_throughputs[key]['prefill_throughput']['mean']

                    group_agg[key]['decode_latency/TPOT_mean'] = self.latancy_throughputs[key]['decode_latency']['mean']
                    group_agg[key]['decode_latency/TPOT_median'] = self.latancy_throughputs[key]['decode_latency']['median']
                    group_agg[key]['decode_latency/TPOT_p99'] = self.latancy_throughputs[key]['decode_latency']['p99']
                    group_agg[key]['decode_throughput'] = self.latancy_throughputs[key]['decode_throughput']['mean']
                    
                    
                    group_agg[key]['total_latency_mean'] = self.latancy_throughputs[key]['total_latency']['mean']
                    group_agg[key]['total_latency_median'] = self.latancy_throughputs[key]['total_latency']['median']
                    group_agg[key]['total_latency_p99'] = self.latancy_throughputs[key]['total_latency']['p99']
                    group_agg[key]['total_throughput'] = self.latancy_throughputs[key]['total_throughput']['mean']
                else:
                    if key not in results_agg.keys():
                        results_agg[key] = {}
                    results_agg[key]['prefill_latency/TTFT_mean'] = self.latancy_throughputs[key]['prefill_latency']['mean']
                    results_agg[key]['prefill_latency/TTFT_median'] = self.latancy_throughputs[key]['prefill_latency']['median']
                    results_agg[key]['prefill_latency/TTFT_p99'] = self.latancy_throughputs[key]['prefill_latency']['p99']
                    results_agg[key]['prefill_throughput'] = self.latancy_throughputs[key]['prefill_throughput']['mean']

                    results_agg[key]['decode_latency/TPOT_mean'] = self.latancy_throughputs[key]['decode_latency']['mean']
                    results_agg[key]['decode_latency/TPOT_median'] = self.latancy_throughputs[key]['decode_latency']['median']
                    results_agg[key]['decode_latency/TPOT_p99'] = self.latancy_throughputs[key]['decode_latency']['p99']
                    results_agg[key]['decode_throughput'] = self.latancy_throughputs[key]['decode_throughput']['mean']
                    
                    results_agg[key]['total_latency_mean'] = self.latancy_throughputs[key]['total_latency']['mean']
                    results_agg[key]['total_latency_median'] = self.latancy_throughputs[key]['total_latency']['median']
                    results_agg[key]['total_latency_p99'] = self.latancy_throughputs[key]['total_latency']['p99']
                    results_agg[key]['total_throughput'] = self.latancy_throughputs[key]['total_throughput']['mean']

            results_dict = {
                "results": dict(results_agg.items()),
                **(
                    {"groups": dict(group_agg.items())}
                    if (bool(group_agg) & show_group_table)
                    else {}
                ),
                "group_subtasks": dict(reversed(subtask_list.items())),
                "configs": dict(sorted(configs.items())),
                "versions": dict(sorted(versions.items())),
                "n-shot": dict(sorted(num_fewshot.items())),
                "higher_is_better": dict(sorted(higher_is_better.items())),
                "n-samples": {
                    task_output.task_name: {
                        "original": len(task_output.task.eval_docs),
                        "effective": min(
                            limit if limit else len(task_output.task.eval_docs),
                            len(task_output.task.eval_docs),
                        ),
                    }
                    for task_output, limit in zip(self.eval_tasks, self.limits)
                },
            }
            if self.log_samples:
                results_dict["samples"] = dict(samples)

            return results_dict

        else:
            return None


    def postprocess_results(self, results):
        if self.model.rank == 0:
            if isinstance(self.model, str):
                model_name = self.model
            elif hasattr(self.model, "config") and hasattr(self.model.config, "_name_or_path"):
                model_name = self.model.config._name_or_path
            else:
                model_name = type(self.model).__name__

            # add info about the model and few shot config
            results["config"] = {
                "model": model_name,
                "model_args": self.model_args,
            }
            # # add more detailed model info if available
            results["config"].update(
                {
                    "batch_size": self.batch_sizes,
                    "batch_sizes": (
                        list(self.model.batch_sizes.values()) if hasattr(self.model, "batch_sizes") else []
                    ),
                    "use_cache": self.use_cache,
                    "limit": self.limit,
                    "bootstrap_iters": self.bootstrap_iters,
                    "gen_kwargs": self.gen_kwargs,
                    "random_seed": self.random_seed,
                    "numpy_seed": self.numpy_random_seed,
                    "torch_seed": self.torch_random_seed,
                    "fewshot_seed": self.fewshot_random_seed,
                }
            )
            results["git_hash"] = get_git_commit_hash()
            results["date"] = self.start_date
            add_env_info(results)  # additional environment info to results
            add_tokenizer_info(results, self.model)  # additional info about tokenizer
            return results
        else:
            return None
    
    def compute_latency_throughput(self, latency_dict: defaultdict[list], throughput_dict: defaultdict[list]):
        """
        计算批量请求的平均、中位和前99百分位的延迟、TPOT和TTFT
        TTFT为首token生成时间, 等价于prefill.latency
        TPOT为decode阶段每两个token生成的间隔时间, 等价于decode.throughput
        
        参数:
        latency_dict: 包含三种延迟数据的字典 (prefill, decode, total)
        throughput_dict: 包含三种吞吐量数据的字典 (prefill, decode, total)
        
        返回:
        包含所有计算指标的字典
        """
        # 验证数据完整性
        required_metrics = ["prefill", "decode", "total"]
        for metric in required_metrics:
            if len(latency_dict.get(metric, [])) == 0:
                raise ValueError(f"No latency data for {metric} to compute statistics.")
            if len(throughput_dict.get(metric, [])) == 0:
                raise ValueError(f"No throughput data for {metric} to compute statistics.")
        
        results = {}
        
        # 计算延迟和吞吐量统计数据
        for key in required_metrics:

            latency_stats = self._calculate_statistics(latency_dict[key], key=key)
            throughput_stats = self._calculate_statistics(throughput_dict[key], key=key)
            results[key+'_latency'] = latency_stats[key]
            results[key+'_throughput'] = throughput_stats[key]
        
        # 统计TTFT和TPOT
        
        return results
    
    def _calculate_statistics(self, values: list, key: str) -> dict:
        """
        辅助方法：计算给定数据的统计指标
        
        参数:
        data_dict: 包含性能数据的字典
        
        返回:
        包含平均值、中位数和99百分位值的统计字典
        """
        stats = {}
        data_array = np.array(values)
            
        # 计算统计指标
        mean_val = np.mean(data_array)
        median_val = np.median(data_array)
        p99_val = np.percentile(data_array, 99)
        
        # 存储结果
        stats[key] = {
            "mean": mean_val,
            "median": median_val,
            "p99": p99_val
        }
        
        return stats    
    # def compute_latency_throughput(self, 
    #                                latancy_dict: defaultdict[list], 
    #                                throughput_dict: defaultdict[list]):
    #     if len(latancy_dict["prefill"]) == 0 or len(latancy_dict["decode"]) == 0 or len(latancy_dict["total"]) == 0:
    #         raise ValueError("No data to compute latency and throughput.")
    #     if len(throughput_dict["prefill"]) == 0 or len(throughput_dict["decode"]) == 0 or len(throughput_dict["total"]) == 0:
    #         raise ValueError("No data to compute latency and throughput.")
        
    #     total_prefill_latency = sum(latancy_dict["prefill"])
    #     total_decode_latency = sum(latancy_dict["decode"])
    #     total_total_latency = sum(latancy_dict["total"])

    #     total_prefill_throughput = sum(throughput_dict["prefill"])
    #     total_decode_throughput = sum(throughput_dict["decode"])
    #     total_total_throughput = sum(throughput_dict["total"])

    #     mean_prefill_latency = total_prefill_latency / len(latancy_dict["prefill"])
    #     mean_decode_latency = total_decode_latency / len(latancy_dict["decode"])
    #     mean_total_latency = total_total_latency / len(latancy_dict["total"])
    #     mean_prefill_throughput = total_prefill_throughput / len(throughput_dict["prefill"])
    #     mean_decode_throughput = total_decode_throughput / len(throughput_dict["decode"])
    #     mean_total_throughput = total_total_throughput / len(throughput_dict["total"])

    #     return [mean_prefill_latency, mean_decode_latency, mean_total_latency, mean_prefill_throughput, mean_decode_throughput, mean_total_throughput]
        




def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args
