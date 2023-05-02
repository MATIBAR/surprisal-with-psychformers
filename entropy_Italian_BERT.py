!pip install transformers
!pip install torch
!transformers-cli download dbmdz/bert-base-italian-cased
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

def calculate_entropy(text):
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')
    model = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-italian-cased')
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, :-1, :]
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log2(probs)
        entropy = -1.0 * torch.sum(probs * log_probs, dim=-1).cpu().numpy()
    return np.mean(entropy)

text = "questa è una frase test"
entropy = calculate_entropy(text)

print(f"Entropy: {entropy}")
