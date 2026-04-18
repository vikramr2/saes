import csv
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

METADATA_CSV = "../data/oc_mini_node_metadata.csv"
OUTPUT_NPY = "../data/embeddings.npy"
OUTPUT_IDS = "../data/embedding_ids.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_texts(path):
    ids, texts = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row["abstract"].strip():
                continue
            ids.append(int(row["id"]))
            texts.append(f"{row['title']}: {row['abstract']}")
    return ids, texts

def encode(texts, tokenizer, model):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings.cpu().float())
    return torch.cat(all_embeddings, dim=0).numpy()

def main():
    print(f"Using device: {DEVICE}")
    print("Loading texts...")
    ids, texts = load_texts(METADATA_CSV)
    print(f"  {len(texts)} texts loaded")

    print("Loading SPECTER2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    embeddings = encode(texts, tokenizer, model)

    np.save(OUTPUT_NPY, embeddings)
    np.save(OUTPUT_IDS, np.array(ids))
    print(f"Saved embeddings {embeddings.shape} → {OUTPUT_NPY}")
    print(f"Saved IDs {len(ids)} → {OUTPUT_IDS}")

if __name__ == "__main__":
    main()
