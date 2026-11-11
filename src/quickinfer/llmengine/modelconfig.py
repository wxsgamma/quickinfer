from dataclasses import dataclass

@dataclass
class ModelConfig:
    model:str
    dtype:str="fp16"
    max_model_len=4096
    temperature:float=0.6
     