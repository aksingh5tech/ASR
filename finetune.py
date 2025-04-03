# pip install transformers datasets torchaudio jiwer


import os
import torch
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("common_voice", "en", split="train[:1%]")  # Example dataset
test_dataset = load_dataset("common_voice", "en", split="test[:1%]")
# Preprocess audio data
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
def preprocess(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch
dataset = dataset.map(preprocess, remove_columns=["audio", "sentence"])
test_dataset = test_dataset.map(preprocess, remove_columns=["audio", "sentence"])
# Load pre-trained model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    bos_token_id=processor.tokenizer.bos_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
)
# Set data collator
def data_collator(features):
    input_values = [f["input_values"] for f in features]
    labels = [f["labels"] for f in features]
    batch = processor.pad({"input_values": input_values, "labels": labels}, return_tensors="pt")
    batch["labels"] = torch.tensor([[-100 if token == processor.tokenizer.pad_token_id else token for token in label]
                                    for label in batch["labels"]])
    return batch
# Define metrics
wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
# Define training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned",
    group_by_length=True,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=3,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)
# Create Trainer instance
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)
# Start training
trainer.train()
# Save the model and processor
model.save_pretrained("./wav2vec2-finetuned")
processor.save_pretrained("./wav2vec2-finetuned")