import torch
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification

# Load the fine-tuned BERT model
model_path = "path/to/saved/model"
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")

# Load the new dataset
new_data = pd.read_csv("path/to/new/dataset.csv")

# Tokenize the texts
tokenized_texts = [tokenizer.encode(text, max_length=512, truncation=True) for text in new_data['text']]

# Pad the sequences to the same length
max_len = max([len(text) for text in tokenized_texts])
padded_texts = [text + [0]*(max_len-len(text)) for text in tokenized_texts]

# Convert the texts to PyTorch tensors
input_ids = torch.tensor(padded_texts)
attention_mask = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))

# Set the device to use the GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Make predictions
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

# Print the predicted labels
print(predictions)
