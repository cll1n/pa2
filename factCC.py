from transformers import BertForSequenceClassification, BertTokenizer
import json

model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


file_path = "testValues.json"

with open(file_path, "r") as file:
    data = json.load(file)

results = []
for item in data:
  input_dict = tokenizer(item['infos'], item['response'], max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
  logits = model(**input_dict).logits
  pred = logits.argmax(dim=1)
  results.append(model.config.id2label[pred.item()])

print(results.count("CORRECT"),  results.count("INCORRECT"))
