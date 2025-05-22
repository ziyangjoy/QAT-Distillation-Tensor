# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast
task="sst2"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained(f"JeremiahZ/bert-base-uncased-{task}")
model = AutoModelForSequenceClassification.from_pretrained(f"JeremiahZ/bert-base-uncased-{task}")

# Evaluate the model
import torch
from datasets import load_dataset
dataset = load_dataset("glue", task)

sentence1_key, sentence2_key = None, None
if task in ["cola", "sst2"]:
    sentence1_key = "sentence"
    sentence2_key = None
elif task in ["mrpc", "qqp",'rte']:
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif task == "stsb":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif task in ["mnli", "mnli-mm", "mnli-m", "qnli", "wnli"]:
    sentence1_key, sentence2_key = "premise", "hypothesis"
else:
    # Try to infer keys
    keys = list(dataset["train"].features.keys())
    keys = [k for k in keys if k != "label"]
    if len(keys) == 1:
        sentence1_key = keys[0]
    elif len(keys) >= 2:
        sentence1_key, sentence2_key = keys[:2]
    else:
        raise ValueError(f"Cannot determine input keys for task {task}")

# Tokenize datasets
def preprocess_function(examples):
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        truncation=True,
        padding="max_length",  # Pad all sequences to the same length
        max_length=128,        # Ensure all sequences are truncated to max_length
    )

encoded_dataset = dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)  # Convert logits to a PyTorch tensor
    predictions = torch.argmax(logits, axis=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean()
    return {"accuracy": accuracy.item()}

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Keep evaluation strategy as "epoch"
    per_device_eval_batch_size=16,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset["validation"],  # Pass the evaluation dataset here
    compute_metrics=compute_metrics,
)

# Evaluate the model
trainer.evaluate()