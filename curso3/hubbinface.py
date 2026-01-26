# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 08:47:39 2025

@author: jesus
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification,  Trainer, TrainingArguments
import torch
import tensorflow as tf
from torch.utils.data import TensorDataset

classifier = pipeline("sentiment-analysis")  # many other tasks are available
result = classifier("The actors were very convincing.")


print(result)

classifier(["I am from India.", "I am from Iraq."])


model_name = "huggingface/distilbert-base-uncased-finetuned-mnli"

classifier_mnli = pipeline("text-classification", model=model_name)
classifier_mnli("She loves me. [SEP] She loves me not.")




tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



token_ids = tokenizer(["I like soccer. [SEP] We all love soccer!",
                       "Joe lived for a very long time. [SEP] Joe is old."],
                      padding=True, return_tensors="pt")


print(token_ids)

token_ids = tokenizer([("I like soccer.", "We all love soccer!"),
                       ("Joe lived for a very long time.", "Joe is old.")],
                      padding=True, return_tensors="pt")

print(token_ids)


with torch.no_grad():
  outputs = model(**token_ids)
print(outputs)


Y_logits = tf.constant(outputs.logits)
print(Y_logits)


Y_probas = tf.keras.activations.softmax(Y_logits)
print(Y_probas)

Y_pred = tf.argmax(Y_probas, axis=1)
print(Y_pred )

sentences = [("Sky is blue", "Sky is red"), ("I love her", "She loves me")]
X_train = tokenizer(sentences, padding=True, return_tensors="tf").data
y_train = tf.constant([0,2])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=2)


sentences = [("Sky is blue", "Sky is red"), ("I love her", "She loves me")]
X_train = tokenizer(sentences, padding=True, return_tensors="pt").data
y_train = torch.tensor([0, 2])  # 0=contradiction, 2=neutral

dataset = TensorDataset(X_train["input_ids"], X_train["attention_mask"],
                        y_train)

def collate_fn(batch):
    input_ids, attention_mask, labels = zip(*batch)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels)
    }

args = TrainingArguments(output_dir="./results", num_train_epochs=2,
                         per_device_train_batch_size=2, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=dataset,
                  data_collator=collate_fn)
trainer.train()





