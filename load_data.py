from datasets import load_dataset
from huggingface_hub import login

 
full_persona = load_dataset(
    "LLM-Digital-Twin/Twin-2K-500",
    "full_persona"
)

wave_split = load_dataset(
    "LLM-Digital-Twin/Twin-2K-500",
    "wave_split"
)

print(full_persona.shape)

data_object = full_persona['data']
print(type(data_object))

print(type(full_persona))
print(full_persona.keys())