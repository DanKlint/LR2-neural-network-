from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from bson import decode_all  # pip install pymongo, pip install bson, python 13?
import os
import random
def normalize(x):
    return x.replace(',', '').replace('.', '').replace('"', '').replace('!', '').replace('\n', '').lower()

def classify_text(text: str):
    encoding = tokenizer(
        normalize(text),
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()
    return 'commerce' if predicted_class == 1 else 'plain text'

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the users.bson file
bson_file_path = os.path.join(script_dir, 'users.bson')

# Read the BSON data from the file
with open(bson_file_path, 'rb') as f:
    users_data = f.read()

users = decode_all(users_data)
random.seed(4)

model = GPT2ForSequenceClassification.from_pretrained('./commerce_classifier')
tokenizer = GPT2Tokenizer.from_pretrained('./commerce_classifier')
tokenizer.pad_token = tokenizer.eos_token
while True:
    random_user = random.choice(users)
    random_user_status = random_user.get("status")
    if random_user_status is None or random_user_status == '':
        continue
    classification = classify_text(random_user_status)
    print(f"Status: {random_user_status}")
    print(f"The text is classified as: {classification}")
    choice = input("Продолжить? Y/N\n")
    if choice == "N":
        break
