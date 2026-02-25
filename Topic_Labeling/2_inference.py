import os
import json
from typing import List, Dict
import argparse
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import transformers

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("-" * 60)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU available:", True)
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print("GPU available:", False)
    print("Using CPU")

CHECKPOINT_PATH = os.path.join("checkpoints_7_epochs", "best_student_model.pt")

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "labeled_CSVs"

BATCH_SIZE = 16
SAVE_EVERY = 800

LOCK_SUFFIX = ".lock"
DONE_SUFFIX = ".done"

CHUNK_STRIDE = 384
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MAX_CHUNKS = 4

class HierarchicalDebertaMultiTask(nn.Module):
    def __init__(self, model_name: str, num_topics: int, attention_size: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.attention_w = nn.Linear(hidden_size, attention_size)
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
        bsz, n_chunks, seq_len = input_ids.size()

        flat_input_ids = input_ids.view(bsz * n_chunks, seq_len)
        flat_attention_mask = attention_mask.view(bsz * n_chunks, seq_len)

        encoder_outputs = self.encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        if hasattr(encoder_outputs, "last_hidden_state"):
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs[0]

        chunk_cls = hidden_states[:, 0, :]
        hidden_size = chunk_cls.size(-1)
        chunk_cls = chunk_cls.view(bsz, n_chunks, hidden_size)

        u = torch.tanh(self.attention_w(chunk_cls))
        scores = self.attention_v(u).squeeze(-1)

        mask = (chunk_mask > 0).float()
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        doc_rep = torch.sum(attn_weights * chunk_cls, dim=1)

        x = self.dropout(doc_rep)
        logits_sig = self.sig_head(x)
        logits_topic = self.topic_head(x)

        return {
            "logits_sig": logits_sig,
            "logits_topic": logits_topic,
        }

def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    max_chunks: int,
    stride: int,
) -> Dict[str, torch.Tensor]:
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
            chunk_with_special = (
                chunk_with_special + [tokenizer.pad_token_id] * pad_length
            )
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

print(f"Loading checkpoint from {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

print("Checkpoint file:", CHECKPOINT_PATH)
print("Topic classes:", ckpt["topic_classes"])
print("Number of topic classes:", len(ckpt["topic_classes"]))

state = ckpt["model_state_dict"]

total_params = 0
total_sum = 0.0
for k, v in state.items():
    total_params += v.numel()
    total_sum += float(v.sum())
print("Total parameters:", total_params)
print("Sum of all weights:", total_sum)

topic_classes = ckpt["topic_classes"]
num_topics = len(topic_classes)

config = ckpt.get("config", {})
model_name = config.get("model_name", "microsoft/deberta-v3-base")
max_seq_len = config.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)
max_chunks = config.get("max_chunks", DEFAULT_MAX_CHUNKS)

print(f"Model name from config: {model_name}")
print(f"Max seq len: {max_seq_len}, max chunks: {max_chunks}")
print(f"Number of topic classes: {num_topics}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = HierarchicalDebertaMultiTask(model_name, num_topics=num_topics)

load_result = model.load_state_dict(state, strict=True)
print("State dict loaded successfully")

model.to(DEVICE)
model.eval()

def predict_batch(texts: List[str]):
    """Run model on a list of texts and return predicted significant label and topic string."""
    batch_input_ids = []
    batch_attention_mask = []
    batch_chunk_mask = []

    for t in texts:
        if not isinstance(t, str):
            t = "" if t is None else str(t)

        chunked = chunk_text(
            text=t,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_chunks=max_chunks,
            stride=CHUNK_STRIDE,
        )

        batch_input_ids.append(chunked["input_ids"])
        batch_attention_mask.append(chunked["attention_mask"])
        batch_chunk_mask.append(chunked["chunk_mask"])

    input_ids = torch.stack(batch_input_ids, dim=0).to(DEVICE)
    attention_mask = torch.stack(batch_attention_mask, dim=0).to(DEVICE)
    chunk_mask = torch.stack(batch_chunk_mask, dim=0).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_mask=chunk_mask,
        )

    logits_sig = outputs["logits_sig"]
    logits_topic = outputs["logits_topic"]

    preds_sig = logits_sig.argmax(dim=-1).cpu().tolist()
    preds_topic_idx = logits_topic.argmax(dim=-1).cpu().tolist()
    preds_topic_label = [topic_classes[i] for i in preds_topic_idx]

    return preds_sig, preds_topic_label

test_text = "This is a short speech about crime, police and public safety."
pred_sig, pred_topic = predict_batch([test_text])
print("Sanity test")
print("  text:", test_text)
print("  predicted significant:", bool(pred_sig[0]))
print("  predicted topic:", pred_topic[0])
print("-" * 60)

def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def process_jsonl_file(jsonl_path: str) -> bool:
    """
    Process a single JSONL file and return True if any new ids were processed,
    False if everything was already up to date or there were no records.
    """
    file_name = os.path.basename(jsonl_path)
    base_name, _ = os.path.splitext(file_name)
    print(f"\nProcessing file: {file_name}")

    items = read_jsonl(jsonl_path)
    n_rows = len(items)
    print(f"{file_name}: total jsonl records = {n_rows}")

    for i in range(min(5, n_rows)):
        item_id = items[i].get("id")
        speaking_val = items[i].get("speaking", "")
        print(f"Example {i} id={item_id}")
        print("speaking snippet:", str(speaking_val)[:200].replace("\n", " "))
        print("-" * 60)

    if n_rows == 0:
        print(f"No records in {file_name}, nothing to do")
        return False

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.csv")

    if os.path.isfile(output_path):
        existing_df = pd.read_csv(output_path)
        if "filename" in existing_df.columns:
            processed_ids = set(existing_df["filename"].astype(str).tolist())
        else:
            processed_ids = set()
        print(
            f"{file_name}: existing CSV found with {len(existing_df)} rows, "
            f"{len(processed_ids)} ids already processed"
        )
    else:
        existing_df = pd.DataFrame(
            columns=["filename", "significant", "topic_extracted"]
        )
        processed_ids = set()
        print(f"{file_name}: no existing CSV, starting fresh")

    new_rows = []
    newly_processed = 0

    total_already = len(processed_ids)

    for start_idx in range(0, n_rows, BATCH_SIZE):

        batch_ids = []
        batch_texts = []

        end_idx = min(start_idx + BATCH_SIZE, n_rows)

        for i in range(start_idx, end_idx):
            item = items[i]
            item_id = str(item.get("id", ""))
            if item_id == "":
                continue

            if item_id in processed_ids:
                continue

            speaking_val = item.get("speaking", "")
            if speaking_val is None:
                speaking_val = ""

            batch_ids.append(item_id)
            batch_texts.append(str(speaking_val))

        if not batch_ids:
            continue

        preds_sig, preds_topic = predict_batch(batch_texts)

        if start_idx == 0:
            print("Debug first batch predictions:")
            for bid, txt, sig, topic in zip(
                batch_ids, batch_texts, preds_sig, preds_topic
            ):
                print(f"  id={bid} significant={bool(sig)} topic={topic}")
                print("    text snippet:", txt[:120].replace("\n", " "))
            print("-" * 60)

        for item_id, sig, topic in zip(batch_ids, preds_sig, preds_topic):
            new_rows.append(
                {
                    "filename": item_id,
                    "significant": bool(sig),
                    "topic_extracted": topic,
                }
            )
            processed_ids.add(item_id)
            newly_processed += 1

        processed_total = total_already + newly_processed
        print(f"{file_name}: processed {processed_total} / {n_rows} records")

        if (newly_processed % SAVE_EVERY == 0) or (start_idx + BATCH_SIZE >= n_rows):
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = existing_df

            combined_df.to_csv(output_path, index=False)
            print(f"{file_name}: intermediate save to {output_path}")

            existing_df = combined_df
            new_rows = []

    if newly_processed == 0:
        print(f"{file_name}: no new ids to process, CSV already up to date")
        return False
    else:
        print(f"Finished {file_name}, final output saved to {output_path}")
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_idx",
        type=int,
        default=None,
        help="Worker id, usually SLURM_ARRAY_TASK_ID",
    )
    return parser.parse_args()

def get_jsonl_files():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' does not exist")
        return []

    jsonl_files = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".jsonl")
    ]
    jsonl_files = sorted(jsonl_files)
    return jsonl_files

def lock_path_for(jsonl_path: str) -> str:
    return jsonl_path + LOCK_SUFFIX

def done_path_for(jsonl_path: str) -> str:
    return jsonl_path + DONE_SUFFIX

def acquire_lock(jsonl_path: str, worker_id: int) -> bool:
    """
    Try to create a lock file for this JSONL.
    Returns True if the lock was acquired by this worker.
    """
    lock_path = lock_path_for(jsonl_path)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"worker {worker_id}\n")
        return True
    except FileExistsError:
        return False

def release_lock(jsonl_path: str):
    lock_path = lock_path_for(jsonl_path)
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass

def worker_loop(worker_id: int):
    """
    Worker loop:
    repeatedly scan the list of JSONL files,
    try to lock one that is not done yet,
    process it,
    and continue until no work is left.
    """
    jsonl_files = get_jsonl_files()
    if not jsonl_files:
        print(f"Worker {worker_id}: no JSONL files found in '{INPUT_FOLDER}'")
        return

    print(f"Worker {worker_id}: found {len(jsonl_files)} JSONL file(s) in '{INPUT_FOLDER}'")

    while True:
        got_work = False

        for jsonl_path in jsonl_files:
            done_flag = done_path_for(jsonl_path)

            if os.path.exists(done_flag):
                continue

            if not acquire_lock(jsonl_path, worker_id):
                continue

            got_work = True
            try:
                print(
                    f"Worker {worker_id}: acquired {os.path.basename(jsonl_path)}, starting processing"
                )
                had_new = process_jsonl_file(jsonl_path)

                open(done_flag, "w").close()
                print(
                    f"Worker {worker_id}: marked {os.path.basename(jsonl_path)} as done "
                    f"(had_new={had_new})"
                )
            finally:
                release_lock(jsonl_path)
                print(
                    f"Worker {worker_id}: released lock for {os.path.basename(jsonl_path)}"
                )

        if not got_work:
            print(f"Worker {worker_id}: no more files to process, exiting")
            break

def main():
    args = parse_args()
    worker_id = args.task_idx if args.task_idx is not None else 0
    worker_loop(worker_id)

if __name__ == "__main__":
    main()
