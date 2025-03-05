# ✅ Step 1: Install Required Libraries
import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from pypdf import PdfReader

# ✅ Step 2: Load All PDFs from "Books/" Folder
def load_pdfs_from_folder(folder_path="Books"):
    all_texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            pdf_text = extract_text_from_pdf(pdf_path)
            all_texts.append(pdf_text)
    return all_texts

# ✅ Step 3: Extract Text from Each PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ✅ Step 4: Tokenization
def tokenize_texts(texts, tokenizer, max_length=512):
    tokenized_texts = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return Dataset.from_dict({"input_ids": tokenized_texts["input_ids"], "labels": tokenized_texts["input_ids"]})

# ✅ Step 5: Load Pretrained Mistral 7B Model with 4-bit Quantization
def load_mistral_model():
    model_name = "mistralai/Mistral-7B-v0.1"

    # Apply 4-bit quantization for faster fine-tuning
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

    return model, tokenizer

# ✅ Step 6: Apply LoRA (Fine-Tuning Optimization)
def apply_lora(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model

# ✅ Step 7: Fine-Tune the Model
def fine_tune_model(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir="./fine_tuned_mistral",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=500,  # Reduce for quick testing
        learning_rate=2.5e-5,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_strategy="steps",
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {"input_ids": torch.tensor([f["input_ids"] for f in data])},
    )

    print("🚀 Training Started...")
    trainer.train()
    print("🎯 Fine-Tuning Completed!")

# ✅ Step 8: Run the Fine-Tuning Pipeline
def main():
    print("📥 Loading PDFs from 'Books' Folder...")
    pdf_texts = load_pdfs_from_folder()

    print("📖 Loading Mistral 7B Model...")
    model, tokenizer = load_mistral_model()

    print("🔠 Tokenizing PDF Data...")
    dataset = tokenize_texts(pdf_texts, tokenizer)

    print("⚡ Applying LoRA for Efficient Fine-Tuning...")
    model = apply_lora(model)

    print("🎯 Fine-Tuning Model on Historical PDFs...")
    fine_tune_model(model, tokenizer, dataset)

    print("✅ Model Fine-Tuned Successfully! Ready for Historical AI Guide!")

# Run the Fine-Tuning Pipeline
if __name__ == "__main__":
    main()
