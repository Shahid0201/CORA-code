import os
import math
import random
import pandas as pd
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

CSV_PATH = "training_data.csv"

MODEL_NAME = "microsoft/deberta-v3-base"

MAX_SEQ_LEN = 512
MAX_CHUNKS = 8
CHUNK_STRIDE = 384

BATCH_SIZE = 4
LR = 2e-5
EPOCHS = 3

WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

df = pd.read_csv(CSV_PATH)

required_cols = {"filename", "Significant", "topic_1", "speaking"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

if df["Significant"].dtype == bool:
    df["Significant"] = df["Significant"].astype(int)
else:

    def map_sig(x):
        if isinstance(x, str):
            x_lower = x.strip().lower()
            if x_lower in {"1", "true", "t", "yes", "y"}:
                return 1
            if x_lower in {"0", "false", "f", "no", "n"}:
                return 0
        try:
            return int(x)
        except Exception:
            return 0
    df["Significant"] = df["Significant"].apply(map_sig)

topic_encoder = LabelEncoder()
df["topic_label"] = topic_encoder.fit_transform(df["topic_1"].astype(str))
num_topics = len(topic_encoder.classes_)

print(f"Number of rows: {len(df)}")
print(f"Number of topic classes: {num_topics}")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["topic_label"],
    random_state=RANDOM_SEED,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    max_chunks: int,
    stride: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise the full text without truncation, then split into chunks
    of length max_seq_len, with overlap controlled by stride.
    Returns tensors of shape (max_chunks, max_seq_len) for input_ids and attention_mask,
    plus a chunk_mask of shape (max_chunks,) indicating which chunks are real.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    input_ids_full: List[int] = encoded["input_ids"]

    if len(input_ids_full) == 0:

        input_ids_full = [tokenizer.pad_token_id]

    chunks_ids = []

    step = max_seq_len - 2
    stride_tokens = stride

    start = 0
    while start < len(input_ids_full) and len(chunks_ids) < max_chunks:
        end = start + step
        chunk_tokens = input_ids_full[start:end]

        chunk_with_special = tokenizer.build_inputs_with_special_tokens(chunk_tokens)

        if len(chunk_with_special) < max_seq_len:
            pad_length = max_seq_len - len(chunk_with_special)
            chunk_with_special = chunk_with_special + [tokenizer.pad_token_id] * pad_length
        else:
            chunk_with_special = chunk_with_special[:max_seq_len]

        chunks_ids.append(chunk_with_special)

        start += stride_tokens

    num_real_chunks = len(chunks_ids)

    if num_real_chunks < max_chunks:
        pad_chunk = [tokenizer.pad_token_id] * max_seq_len
        for _ in range(max_chunks - num_real_chunks):
            chunks_ids.append(pad_chunk)

    input_ids_tensor = torch.tensor(chunks_ids, dtype=torch.long)
    attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()

    chunk_mask = torch.zeros(max_chunks, dtype=torch.float)
    chunk_mask[:num_real_chunks] = 1.0

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "chunk_mask": chunk_mask,
    }

class LegislativeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["speaking"])
        sig_label = int(row["Significant"])
        topic_label = int(row["topic_label"])

        chunked = chunk_text(
            text=text,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            max_chunks=MAX_CHUNKS,
            stride=CHUNK_STRIDE,
        )

        sample = {
            "input_ids": chunked["input_ids"],
            "attention_mask": chunked["attention_mask"],
            "chunk_mask": chunked["chunk_mask"],
            "label_sig": torch.tensor(sig_label, dtype=torch.long),
            "label_topic": torch.tensor(topic_label, dtype=torch.long),
        }
        return sample

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    chunk_mask = torch.stack([b["chunk_mask"] for b in batch], dim=0)
    label_sig = torch.stack([b["label_sig"] for b in batch], dim=0)
    label_topic = torch.stack([b["label_topic"] for b in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "label_sig": label_sig,
        "label_topic": label_topic,
    }

train_dataset = LegislativeDataset(train_df)
val_dataset = LegislativeDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

class HierarchicalDebertaForMultiTask(nn.Module):
    def __init__(self, model_name: str, num_topics: int, attention_size: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.attention_w = nn.Linear(hidden_size, attention_size, bias=True)
        self.attention_v = nn.Linear(attention_size, 1, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.sig_head = nn.Linear(hidden_size, 2)
        self.topic_head = nn.Linear(hidden_size, num_topics)

    def forward(
        self,
        input_ids,
        attention_mask,
        chunk_mask,
        label_sig=None,
        label_topic=None,
    ):
        """
        input_ids: (batch, max_chunks, seq_len)
        attention_mask: (batch, max_chunks, seq_len)
        chunk_mask: (batch, max_chunks)
        """
        bsz, n_chunks, seq_len = input_ids.size()

        flat_input_ids = input_ids.view(bsz * n_chunks, seq_len)
        flat_attention_mask = attention_mask.view(bsz * n_chunks, seq_len)

        encoder_outputs = self.encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        hidden_states = encoder_outputs.last_hidden_state

        chunk_cls = hidden_states[:, 0, :]
        hidden_size = chunk_cls.size(-1)

        chunk_cls = chunk_cls.view(bsz, n_chunks, hidden_size)

        u = torch.tanh(self.attention_w(chunk_cls))

        scores = self.attention_v(u).squeeze(-1)

        mask = (chunk_mask > 0).float()
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=1)

        attn_weights = attn_weights.unsqueeze(-1)
        doc_rep = torch.sum(attn_weights * chunk_cls, dim=1)

        x = self.dropout(doc_rep)
        logits_sig = self.sig_head(x)
        logits_topic = self.topic_head(x)

        outputs = {
            "logits_sig": logits_sig,
            "logits_topic": logits_topic,
        }

        loss = None
        if label_sig is not None and label_topic is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_sig = loss_fct(logits_sig, label_sig)
            loss_topic = loss_fct(logits_topic, label_topic)

            loss = loss_sig + loss_topic
            outputs["loss"] = loss
            outputs["loss_sig"] = loss_sig
            outputs["loss_topic"] = loss_topic

        return outputs

model = HierarchicalDebertaForMultiTask(MODEL_NAME, num_topics=num_topics)
model.to(DEVICE)

no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)

num_training_steps = EPOCHS * len(train_loader)
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

def compute_accuracy(preds, labels):
    preds_flat = preds.argmax(dim=-1)
    correct = (preds_flat == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

def train_one_epoch(epoch: int):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} training")

    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        chunk_mask = batch["chunk_mask"].to(DEVICE)
        label_sig = batch["label_sig"].to(DEVICE)
        label_topic = batch["label_topic"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_mask=chunk_mask,
            label_sig=label_sig,
            label_topic=label_topic,
        )

        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (pbar.n + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    total_loss = 0.0
    total_sig_acc = 0.0
    total_topic_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            chunk_mask = batch["chunk_mask"].to(DEVICE)
            label_sig = batch["label_sig"].to(DEVICE)
            label_topic = batch["label_topic"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunk_mask=chunk_mask,
                label_sig=label_sig,
                label_topic=label_topic,
            )

            loss = outputs["loss"]
            logits_sig = outputs["logits_sig"]
            logits_topic = outputs["logits_topic"]

            sig_acc = compute_accuracy(logits_sig, label_sig)
            topic_acc = compute_accuracy(logits_topic, label_topic)

            total_loss += loss.item()
            total_sig_acc += sig_acc
            total_topic_acc += topic_acc
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_sig_acc = total_sig_acc / n_batches
    avg_topic_acc = total_topic_acc / n_batches

    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Validation Significant accuracy: {avg_sig_acc:.4f}")
    print(f"Validation topic accuracy: {avg_topic_acc:.4f}")

    return avg_loss, avg_sig_acc, avg_topic_acc

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    print(f"Epoch {epoch} train loss: {train_loss:.4f}")

    val_loss, val_sig_acc, val_topic_acc = evaluate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("checkpoints", exist_ok=True)
        save_path = os.path.join("checkpoints", "best_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "topic_classes": topic_encoder.classes_.tolist(),
                "config": {
                    "model_name": MODEL_NAME,
                    "max_seq_len": MAX_SEQ_LEN,
                    "max_chunks": MAX_CHUNKS,
                },
            },
            save_path,
        )
        print(f"Saved best model to {save_path}")

print("Training complete")
