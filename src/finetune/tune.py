"""
Finetune an LLM using the generated datasets
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "codellama/CodeLlama-7b-Python-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = [
    "Try solving this programming problem:  Write a function in Python to generate the Fibonacci numbers."
]
input_tokens = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_tokens, max_length=100, do_sample=True, top_p=0.95, top_k=60)
print(tokenizer.decode(output[0], skip_special_tokens=True))
