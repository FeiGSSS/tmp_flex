# from .opt import OPTInputEmbed, OPTTransformerLayer, OPTOutputEmbed
# from .llama import LLaMAInputEmbed, LLaMATransformerLayer
from . import (
    opt, 
    # llama,
    llama_clean,
)
from flexgen.models.config import FlexModelConfig
from flexgen.models.utils import Policy, ExecutionEnv
def get_model_architecture(config:FlexModelConfig, 
                           path: str, 
                           policy:Policy, 
                           env:ExecutionEnv, 
                           weight_map:dict):
# def get_model_architecture(config, 
#                            path, 
#                            policy, 
#                            env, 
#                            weight_map):
    """
    根据 model_type 返回合适的模型层类。

    Returns:
        A class of ModelArchitecture
    """
    model_type = config.model_type
    if model_type in ["opt"]:
        return opt.OptModel(
            config=config,
            path=path,
            policy=policy,
            env=env,
            weight_map=weight_map
        ) 
    elif model_type in ["llama", "mistral"]:   
        return llama_clean.LLaMAModel(
            config=config,
            path=path,
            policy=policy,
            env=env,
            weight_map=weight_map
        )
    elif model_type in ["qwen2"]:
        raise NotImplementedError(f"Model architecture for type '{model_type}' is registering, Please hold on.")
    elif model_type in ["deepseek_V3"]:
        raise NotImplementedError(f"Model architecture for type '{model_type}' is registering, Please hold on.")
    else:
        raise NotImplementedError(f"Model architecture for type '{model_type}' is not registered.")