import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import os 

os.environ["WANDB_API_KEY"] = "e2ab1b2b4244272268524960c98f9a9e6a5decd6"
os.environ["WANDB_PROJECT"]="ft"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

modelpath=r"/tsukimi/llm/Meta-Llama-3-8B"

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)   

dataset = load_dataset("Amirkid/MedQuad-dataset")
all_data = []
for i in range(0,len(dataset["train"]),2):
    all_data.append(f'Question:\n{dataset["train"][i]["text"]} \n\nAnswer:\n{dataset["train"][i+1]["text"]}')
    # all_data.append(f'<|begin_of_text|> {dataset["train"][i]["text"]} [SEP] {dataset["train"][i+1]["text"]} <|end_of_text|>')

dataset_copy = Dataset.from_dict({"text": all_data})
tokenizer.pad_token = tokenizer.eos_token

dataset_tokenized = dataset_copy.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512),
    batched=True, 
    num_proc=1,    # multithreaded
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)
print("Dataset tokenized")

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
    lora_dropout=0.1, 
    bias="none", 
    modules_to_save = ["lm_head", "embed_tokens"],		# needed because we added new tokens to tokenizer/model
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.config.use_cache = False

dataset_tokenized = dataset_tokenized.train_test_split(test_size=0.1)

def collate(elements):
    tokenlist=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokenlist])

    input_ids,labels,attention_masks = [],[],[]
    for tokens in tokenlist:
        pad_len=tokens_maxlen-len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0
        input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
        labels.append( tokens + [-100]*pad_len )    
        attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

    batch={
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks)
    }
    return batch

bs=16      # batch size
ga_steps=1  # gradient acc. steps
epochs=5
steps_per_epoch=len(dataset_tokenized["train"])//(bs*ga_steps)

args = TrainingArguments(
    output_dir="/tsukimi/llm/ft",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch,		# eval and save once per epoch  	
    save_steps=steps_per_epoch,
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=0.00005,
    group_by_length=True,
    fp16=True,
    ddp_find_unused_parameters=False,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    args=args,
)

trainer.train()

model = model.merge_and_unload()
model.save_pretrained("/tsukimi/llm/ft/output")


