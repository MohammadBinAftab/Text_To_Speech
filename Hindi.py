import os
import zipfile
import numpy as np
import pandas as pd
from datasets import Dataset, Audio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from speechbrain.pretrained import EncoderClassifier

# Paths for local data
data_dir = './data'
wav_folder = f'{data_dir}/Hindi_F/wav'
txt_folder = f'{data_dir}/Hindi_F/txt'

# Create necessary directories
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Extract ZIP if not already extracted
if not os.path.exists(wav_folder):
    with zipfile.ZipFile('hindi-f.zip', 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Dataset extracted")

# Load transcripts and wav file paths
file_paths = []
transcripts = []
for file_name in os.listdir(wav_folder):
    if file_name.endswith('.wav'):
        txt_file_name = file_name.replace('.wav', '.txt')
        wav_file_path = os.path.join(wav_folder, file_name)
        txt_file_path = os.path.join(txt_folder, txt_file_name)
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as f:
                transcript = f.read().strip()
            file_paths.append(wav_file_path)
            transcripts.append(transcript)

# Create a dataset
data_dict = {'file_path': file_paths, 'transcript': transcripts}
dataset = Dataset.from_dict(data_dict)
dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))

# Load the T5 processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Speaker embedding model
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speaker_model = EncoderClassifier.from_hparams(source=spk_model_name, run_opts={"device": device})

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

# Prepare dataset
def prepare_dataset(example):
    audio = example["file_path"]
    example = processor(text=example["transcript"], audio_target=audio["array"], sampling_rate=audio["sampling_rate"], return_attention_mask=False)
    example["labels"] = example["labels"][0]
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
    return example

dataset = dataset.map(prepare_dataset)

# Data collator for padding
class TTSDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]
        batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")
        batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
        del batch["decoder_attention_mask"]
        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

# Train/test split
dataset = dataset.train_test_split(test_size=0.1)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./tts_output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=10,
    fp16=True if torch.cuda.is_available() else False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    report_to=["tensorboard"],
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

# Train the model
trainer.train()

# Push model to hub if needed
trainer.push_to_hub("hindi_text_to_speech_tts")

print("Training complete!")
