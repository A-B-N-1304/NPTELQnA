import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, concatenate_datasets, Dataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import streamlit as st

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "google/flan-t5-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 3e-4

# ---------------------------
# LOAD TOKENIZER AND INIT MODEL
# ---------------------------
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

# ---------------------------
# LOAD AND FLATTEN NATURAL QUESTIONS
# ---------------------------
def load_flattened_nq(max_samples=50):
    raw_nq = load_dataset("nq_open", split=f"train[:{max_samples}]")
    contexts, questions, answers = [], [], []
    for item in raw_nq:
        question_data = item.get("question", {})
        question = question_data.get("text", "") if isinstance(question_data, dict) else ""
        annotations = item.get("annotations", [])
        short_answers = []
        if isinstance(annotations, list) and annotations and isinstance(annotations[0], dict):
            short_answers = annotations[0].get("short_answers", [])
        if short_answers:
            answer_text = short_answers[0].get("text", "")
            document = item.get("document", {})
            context = document.get("html", "")
            contexts.append(context)
            questions.append(question)
            answers.append(answer_text)
    return Dataset.from_dict({"context": contexts, "question": questions, "answer": answers})

# ---------------------------
# STAGE 1: PRETRAINING FUNCTION
# ---------------------------
def stage1_train():
    dataset_names = ["quoref", "qasc"]
    datasets = []

    for name in dataset_names:
        try:
            if name == "quoref":
                ds = load_dataset("allenai/quoref", split="train[:150]")
                ds = ds.map(lambda x: {
                    "context": x.get("context", ""),
                    "question": x.get("question", ""),
                    "answer": x.get("answers", {}).get("text", [""])[0]
                })
            elif name == "qasc":
                ds = load_dataset("allenai/qasc", split="train[:150]")
                ds = ds.map(lambda x: {
                    "context": x.get("fact1", ""),
                    "question": x.get("question", ""),
                    "answer": x.get("answerKey", "")
                })
            ds = ds.remove_columns([col for col in ds.column_names if col not in ["context", "question", "answer"]])
            datasets.append(ds)
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    nq_flat = load_flattened_nq(max_samples=100)
    datasets.append(nq_flat)
    combined_dataset = concatenate_datasets(datasets)

    tokenized = combined_dataset.map(preprocess)
    tokenized.set_format(type="torch")
    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        print(f"[Stage 1] Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    model.save_pretrained("t5_stage1_textqa")
    tokenizer.save_pretrained("t5_stage1_textqa")
    print("[✓] Stage 1 model saved to ./t5_stage1_textqa")

# ---------------------------
# STAGE 2: CUSTOM FINETUNING
# ---------------------------
def load_custom_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    contexts, questions, answers = [], [], []
    for item in data:
        context = item.get("context", "")
        for qa in item.get("qa_pairs", []):
            questions.append(qa.get("question", ""))
            answers.append(qa.get("answer", ""))
            contexts.append(context)
    return Dataset.from_dict({"context": contexts, "question": questions, "answer": answers})

def preprocess(example):
    input_text = f"question: {example['question']} context: {example['context']}"
    target_text = example['answer']
    input_enc = tokenizer(input_text, padding="max_length", truncation=True, max_length=200, return_tensors="pt")
    target_enc = tokenizer(target_text, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    return {
        "input_ids": input_enc["input_ids"].squeeze(),
        "attention_mask": input_enc["attention_mask"].squeeze(),
        "labels": target_enc["input_ids"].squeeze()
    }

def train_model(dataset):
    tokenized = dataset.map(preprocess)
    tokenized.set_format(type="torch")
    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        print(f"[Stage 2] Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    model.save_pretrained("t5_stage2_customqa")
    tokenizer.save_pretrained("t5_stage2_customqa")
    print("[✓] Custom model saved to ./t5_stage2_customqa")

# ---------------------------
# INFERENCE
# ---------------------------
def answer_question(video_id, question, data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if item['video_id'] == video_id:
            context = item['context']
            break
    else:
        return "Context for video ID not found."

    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    outputs = model.generate(input_ids, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("\U0001F393 NPTEL QA System")
st.markdown("Ask questions from NPTEL videos.")

video_id = st.text_input("Enter Video ID:")
question = st.text_input("Enter your question:")

if st.button("Get Answer") and video_id and question:
    answer = answer_question(video_id, question, "custom_dataset.json")
    st.success(f"Answer: {answer}")