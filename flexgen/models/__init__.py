from .opt import OPTInputEmbed, OPTTransformerLayer, OPTOutputEmbed
from .llama import LLaMAInputEmbed, LLaMATransformerLayer

def get_model_architecture(model_type: str):
    """
    根据 model_type 返回合适的模型层类。

    Returns:
        A tuple of (InputEmbedClass, TransformerLayerClass, OutputLayerClass)
    """
    if model_type == "opt":
        # 注意: 这里的 OutputLayerClass 需要您单独实现
        return OPTInputEmbed, OPTTransformerLayer, OPTOutputEmbed # Placeholder for OutputLayer
    elif model_type in ["llama", "deepseek", "qwen2", "mistral"]:
        # 注意: 这里的 OutputLayerClass 需要您单独实现
        return LLaMAInputEmbed, LLaMATransformerLayer, None # Placeholder for OutputLayer
    else:
        raise NotImplementedError(f"Model architecture for type '{model_type}' is not registered.")