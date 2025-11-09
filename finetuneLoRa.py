import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import DatasetDict
from dataset2 import HumanVsMachineDataset  # your updated dataset class

# ---- SETTINGS ----
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./lora_qwen_tagged"
CSV_PATH = "ground_truth.csv"
MAX_LEN = 512

# ---- LOAD DATA ----
builder = HumanVsMachineDataset(CSV_PATH)
dataset_dict = builder.load_dataset()
formatted = builder.format_for_training(dataset_dict, few_shot_examples=None, test_mode=False)

train_ds = formatted["train"]
val_ds = formatted["validation"]

# ---- TOKENIZER ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- MAPPING FUNCTION ----
def map_label_to_arabic(label):
    label_map = {
        "human": "بشري",
        "machine": "آلة",
        "Human": "بشري",
        "Machine": "آلة",
        "بشري": "بشري",
        "مولد": "آلة"
    }
    return label_map.get(label.strip(), label.strip())

# ---- TOKENIZE FUNCTION ----
def tokenize_fn(example):
    # Compute max prompt length
    ANSWER_MAX_LEN = 10  # since Arabic labels are 2-3 tokens, 10 is safe
    PROMPT_MAX_LEN = MAX_LEN - ANSWER_MAX_LEN

    # Tokenize the prompt (everything except the answer)
    prompt_enc = tokenizer(example["text"], truncation=True, max_length=PROMPT_MAX_LEN, padding="max_length")


    # Extract label from the text (<answer>بشري</answer>)
    # We assume the label is always at the end after <answer> and </answer>
    if "<answer>" in example["text"]:
        label_text = example["text"].split("<answer>")[-1].split("</answer>")[0]
    else:
        label_text = ""

    # Map to Arabic (just in case)
    label_text = map_label_to_arabic(label_text)
    label_enc = tokenizer(label_text, add_special_tokens=False)
    label_ids = label_enc["input_ids"]

    # ---- CREATE INPUTS AND LABELS ----
    input_ids = prompt_enc["input_ids"] + label_ids
    labels = [-100]*len(prompt_enc["input_ids"]) + label_ids

    # Truncate/pad to MAX_LEN
    input_ids = input_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]
    attention_mask = [1]*len(input_ids)
    while len(input_ids) < MAX_LEN:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        labels.append(-100)
        
    print("=== DEBUG ===")
    print("Prompt length:", len(prompt_enc["input_ids"]))
    print("Answer text:", label_text)
    print("Answer token ids:", label_ids)
    print("Labels (last 10):", labels[-10:])
    print("Input IDs (last 10):", input_ids[-10:])
    print("Attention mask (last 10):", attention_mask[-10:])
    print("================")

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_ds = train_ds.map(tokenize_fn, batched=False)
val_ds = val_ds.map(tokenize_fn, batched=False)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---- LOAD MODEL ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# ---- CONFIGURE LORA ----
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Qwen
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ---- TRAIN ----
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    num_train_epochs=1,
    learning_rate=1e-5,  # lower LR for safety
    logging_steps=10,
    save_strategy="steps",
    output_dir=OUTPUT_DIR,
    fp16=True,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

# ---- SAVE ADAPTER ----
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA adapter saved to", OUTPUT_DIR)
