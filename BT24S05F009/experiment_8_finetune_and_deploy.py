"""
Experiment 8: Fine-tune a small open-source language model with LoRA and deploy it
as a local web application using Gradio.

This is a minimal instruction-tuning example. It uses a tiny GPT-2 model so that
the workflow is easier to understand; quality will be limited. A GPU is recommended.

Install:
    pip install -U torch transformers datasets peft accelerate gradio

Run:
    python experiment_8_finetune_and_deploy.py

The script trains a LoRA adapter, saves it in ./lora_adapter, then starts a local
Gradio interface. The first run downloads the base model.
"""

import torch
import gradio as gr
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

BASE_MODEL = "sshleifer/tiny-gpt2"
OUTPUT_DIR = "lora_adapter"

TRAINING_TEXTS = [
    "Question: What is Python? Answer: Python is a popular programming language.",
    "Question: What is machine learning? Answer: Machine learning allows computers to learn patterns from data.",
    "Question: What is RAG? Answer: RAG retrieves relevant information and provides it to a language model.",
    "Question: What is a neural network? Answer: A neural network is a model inspired by connected neurons.",
    "Question: What is generative AI? Answer: Generative AI creates new content such as text, images, or audio."
]

def tokenize_dataset(tokenizer):
    dataset = Dataset.from_dict({"text": TRAINING_TEXTS})
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=96)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized

def train_adapter():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["c_attn"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    tokenized = tokenize_dataset(tokenizer)
    args = TrainingArguments(
        output_dir="training_checkpoints",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved in ./{OUTPUT_DIR}")

def load_deployed_model():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    model.eval()
    return tokenizer, model

def answer(question, tokenizer, model):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    import os
    if not os.path.exists(OUTPUT_DIR):
        train_adapter()

    tokenizer, model = load_deployed_model()
    demo = gr.Interface(
        fn=lambda question: answer(question, tokenizer, model),
        inputs=gr.Textbox(lines=2, label="Ask a question"),
        outputs=gr.Textbox(lines=5, label="Model response"),
        title="Fine-tuned Generative AI Model",
        description="A small LoRA fine-tuned language model running locally."
    )
    demo.launch()

if __name__ == "__main__":
    main()
