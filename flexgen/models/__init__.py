from flexgen.models import llama, deepseek
from flexgen.models.config import FlexModelConfig
from flexgen.models.utils import Policy, ExecutionEnv


# def get_model_architecture(config:FlexModelConfig, 
#                            path: str, 
#                            policy:Policy, 
#                            env:ExecutionEnv, 
#                            weight_map:dict):
#     """
#     根据 model_type 返回合适的模型层类。

#     Returns:
#         A class of ModelArchitecture
#     """
#     model_type = config.model_type
#     if model_type in ["opt"]:
#         return opt.OptModel(
#             config=config,
#             path=path,
#             policy=policy,
#             env=env,
#             weight_map=weight_map
#         ) 
#     elif model_type in ["llama"]:   
#         return llama.LLaMAModel(
#             config=config,
#             path=path,
#             policy=policy,
#             env=env,
#             weight_map=weight_map
#         )
#     elif model_type in ["qwen2"]:
#         raise NotImplementedError(f"Model architecture for type '{model_type}' is registering, Please hold on.")
#     elif model_type in ["deepseek_V3"]:
#         raise NotImplementedError(f"Model architecture for type '{model_type}' is registering, Please hold on.")
#     else:
#         raise NotImplementedError(f"Model architecture for type '{model_type}' is not registered.")
    
    
    
def get_model_architecture(pretrained_model_path: str,
                           env:ExecutionEnv, 
                           policy:Policy):
    _path = pretrained_model_path.lower()
    if 'llama-2-7b' in _path:
        return llama.LLaMAModel(
            pretrained_model_path,
            env,
            policy
        )
    elif 'deepseek' in _path:
        return deepseek.DeepSeekV2LiteModel(
            pretrained_model_path=pretrained_model_path,
            env=env,
            policy=policy
        )
    elif 'qwen2' in _path:
        raise NotImplementedError(f"Model architecture for type 'qwen2' is registering, Please hold on.")
    elif 'deepseek_v3' in _path:
        raise NotImplementedError(f"Model architecture for type 'deepseek_v3' is registering, Please hold on.")
    else:
        raise NotImplementedError(f"Model architecture for type 'unknown' is not registered.")