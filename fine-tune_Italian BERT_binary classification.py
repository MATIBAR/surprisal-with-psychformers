import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-italian-cased', num_labels=2)
model.to(device)

# Load data
df = pd.read_csv('data.csv')
texts = df['text'].values.tolist()
labels = df['label'].values.tolist()

# Tokenize and encode the texts
encoded_texts = [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=128, truncation=True) for text in texts]

# Split the data into training and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(encoded_texts, labels, random_state=42, test_size=0.2)

# Convert the inputs to PyTorch tensors
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)

# Set the batch size
batch_size = 16

# Create the DataLoader for training set
train_data = torch.utils.data.TensorDataset(train_inputs, train_labels)
train_sampler = torch.utils.data.RandomSampler(train_data)
train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for validation set
validation_data = torch.utils.data.TensorDataset(validation_inputs, validation_labels)
validation_sampler = torch.utils.data.SequentialSampler(validation_data)
validation_dataloader = torch.utils.data.DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Set the learning rate and optimizer
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Set the number of epochs
num_epochs = 5

# Train the model
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch_inputs = batch[0].to(device)
        batch_labels = batch[1].to(device)
        model.zero_grad()
        loss, _ = model(batch_inputs, token_type_ids=None, attention_mask=(batch_inputs > 0), labels=batch_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.empty_cache()
        if step % 100 == 0:
            print(f"Epoch: {epoch + 1}/{num_epochs}, Step: {step}, Loss: {total_loss / (step + 1)}")
            
# Save the fine-tuned model
model_path = 'fine_tuned_model'
model.save_pretrained(model_path)
