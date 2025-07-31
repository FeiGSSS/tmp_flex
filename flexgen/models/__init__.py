# from .opt import OPTInputEmbed, OPTTransformerLayer, OPTOutputEmbed
# from .llama import LLaMAInputEmbed, LLaMATransformerLayer
from . import (
    opt, 
    llama, 
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
    if model_type == "opt":
        return opt.OptModel(
            config=config,
            path=path,
            policy=policy,
            env=env,
            weight_map=weight_map
        ) 
    elif model_type in ["llama", "deepseek", "qwen2", "mistral"]:   
        return llama 
    else:
        raise NotImplementedError(f"Model architecture for type '{model_type}' is not registered.")