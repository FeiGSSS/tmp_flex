# ==============================================================================
# FILE: auto_flex_model.py
# 描述: 提供给用户的最终接口，封装了所有复杂性。
# ==============================================================================
import os
import pickle
import json
from pathlib import Path
from flexgen.models.load_convert.convert_weights import convert

class AutoFlexModel:
    
    @staticmethod
    def from_pretrained(model_path: str, force_convert: bool = False, **kwargs):
        """
        统一加载入口。自动处理转换和加载。
        
        Args:
            model_path (str): 模型路径，可以是HF模型目录或单个GGUF文件。
            policy: FlexGen的执行策略对象。
            env: FlexGen的执行环境对象。
            force_convert (bool): 是否强制重新转换权重。
        """
        model_name = Path(model_path).name.replace(".gguf", "")
        converted_path = Path(model_path) / f"{model_name}-flexgen"
        
        # 1. 检查是否已转换，如果未转换或强制转换，则执行
        config_path = converted_path / "flexgen_config.pkl"
        if force_convert or not config_path.exists():
            print(f"FlexGen-NumPy weights not found or force_convert=True.")
            print(f"Converting weights from '{model_path}' to '{converted_path}'...")
            os.makedirs(converted_path, exist_ok=True)
            # 注意：这里的 convert 函数需要您在项目中正确导入
            convert(model_path, str(converted_path))
            print("Conversion complete.")
            
        # 2. 加载已转换的 FlexModelConfig
        print(f"Loading FlexGen config from {config_path}")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        # 3. 加载已转换的 FlexModelConfig
        weight_map_path = converted_path / "weight_map.json"
        print(f"Loading FlexGen config from {weight_map_path}")
        with open(weight_map_path, "r") as f:
            weight_map = json.load(f)
            
        return config, str(converted_path), weight_map # 返回配置和路径作为演示