from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from dotenv import load_dotenv

load_dotenv()

model_name = "meta-llama/Llama-2-7b-chat-hf"

class HFModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
    
    def generate_response(self, prompt: str, max_length: int = 50, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
