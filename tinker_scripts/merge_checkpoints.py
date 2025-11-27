from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# Load the base model with the correct class for causal language modeling
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")

# Load the PEFT adapter
model = PeftModel.from_pretrained(
    model, 
    "<downloaded_lora_checkpoint_path>"
)

# Merge the adapter weights with the base model
model = model.merge_and_unload()

# Save the merged model
model.save_pretrained("<merged_checkpoint_path>")
tokenizer.save_pretrained("<merged_checkpoint_path>")

print("Model saved to <merged_checkpoint_path>")