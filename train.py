import numpy as np
import json
from tqdm import tqdm
from jsonargparse import auto_cli

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models import MiniLMSentenceTransformer


def setSeed(seed = 42):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setSeed()


def load_data(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            datas.append(json.loads(line))
    return datas


def info_nce_loss(q_emb, p_emb, n_emb, temperature=0.05):
    q_emb = F.normalize(q_emb, dim=-1)
    p_emb = F.normalize(p_emb, dim=-1)
    n_emb = F.normalize(n_emb, dim=-1)

    pos_sim = (q_emb * p_emb).sum(dim=-1) / temperature
    neg_sim = (q_emb * n_emb).sum(dim=-1) / temperature
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(q_emb.size(0), dtype=torch.long, device=q_emb.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def main(
        model_name: str = None,
        data_path: str = None,
        save_path: str = None,
        batch_size: int = None,
        epochs: int = None,
        max_len: int = None,
        lr: float = None,
        device: str = None
):
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    
    model = MiniLMSentenceTransformer(vocab_size=tokenizer.vocab_size)
    model.to(device)

    datas = load_data(data_path)
    dataloader = DataLoader(datas, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        if epochs == 1:
            pbar = tqdm(dataloader)
        else:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:02d} / {epochs:02d} >>>")
        for batch in pbar:
            queries = batch["query"]
            positives = batch["positive"]
            negatives = batch["negative"]

            q_enc = tokenizer(
                queries,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            p_enc = tokenizer(
                positives,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            n_enc = tokenizer(
                negatives,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            q_emb = model(q_enc["input_ids"], q_enc["attention_mask"])
            p_emb = model(p_enc["input_ids"], p_enc["attention_mask"])
            n_emb = model(n_enc["input_ids"], n_enc["attention_mask"])

            loss = info_nce_loss(q_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Average loss = {avg_loss:.4f}")

        model_revision = f"{model_name}_E{epoch+1}"
        model.set_revision(model_revision)
        torch.save(model.state_dict(), f"{save_path}/{model_revision}.pt")


if __name__ == '__main__':
    auto_cli(main)
