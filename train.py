import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch.nn as nn
from sklearn.metrics import accuracy_score
from transformers import get_scheduler
from preprocess import load_and_prepare_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SentimixtureNet(nn.Module):
    def __init__(self):
        super(SentimixtureNet, self).__init__()
        self.base = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.routing = nn.Linear(768, 768)
        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        routed = torch.relu(self.routing(sequence_output))
        attended, _ = self.attn(routed, routed, routed)
        pooled = attended[:, 0, :]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {'logits': logits, 'loss': loss} if loss is not None else {'logits': logits}

def train():
    train_texts, val_texts, train_labels, val_labels = load_and_prepare_dataset("urdu_sarcastic_dataset.csv")
    print("Label distribution in training set: ")
    print(pd.Series(train_labels).value_counts())

    train_dataset = SarcasmDataset(train_texts, train_labels)
    val_dataset = SarcasmDataset(val_texts, val_labels)

    model = SentimixtureNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    best_acc = 0
    for epoch in range(6):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} - Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs("best_model", exist_ok=True)
            torch.save(model.state_dict(), "best_model/sentimixture_model.pt")
            tokenizer.save_pretrained("best_model")
            print("✅ Best model saved.")

if __name__ == "__main__":
    import pandas as pd
    train()