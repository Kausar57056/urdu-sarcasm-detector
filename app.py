import streamlit as st
import torch
import torch.nn as nn
import os
import requests
from transformers import AutoTokenizer, XLMRobertaModel
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# ------------------------------
# SentimixtureNet Model Class
# ------------------------------
class SentimixtureNet(nn.Module):
    def __init__(self):
        super(SentimixtureNet, self).__init__()
        self.base = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.routing = nn.Linear(768, 768)
        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        routed = torch.relu(self.routing(sequence_output))
        attended, _ = self.attn(routed, routed, routed)
        pooled = attended[:, 0, :]
        logits = self.classifier(pooled)
        return logits

# ------------------------------
# Dropbox Download Function
# ------------------------------
def download_from_dropbox(url, local_path):
    try:
        if not os.path.exists(local_path):
            r = requests.get(url.replace("?dl=0", "?dl=1"), stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"❌ Error downloading {local_path}: {e}")
        return False

# ------------------------------
# Load Model and Tokenizer (Cached)
# ------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model_url = "https://www.dropbox.com/scl/fi/6p5l4kmjwglsx4bzp7u9t/sentimixture_model.pt?rlkey=qyii1kgniduebx4qyjagim0xk&st=pmmh1dut&dl=1"
    model_path = "sentimixture_model.pt"
    tokenizer_dir = "tokenizer/"
    os.makedirs(tokenizer_dir, exist_ok=True)

    tokenizer_files = {
        "sentencepiece.bpe.model": "https://www.dl.dropboxusercontent.com/scl/fi/zb8h3hpt2g7glopn2xhqx/sentencepiece.bpe.model?rlkey=eqwaym45sg419b5m67zxs0jfs&st=97sy243j&raw=1",
        "special_tokens_map.json": "https://www.dl.dropboxusercontent.com/scl/fi/ehiwmv4wwiveto6lqleac/special_tokens_map.json?rlkey=q4gpcw7z9nwk97cmlgxrb52ky&st=zng6v2qx&raw=1",
        "tokenizer_config.json": "https://www.dl.dropboxusercontent.com/scl/fi/4kv21xx93zqs6koarzvca/tokenizer_config.json?rlkey=5jdqzmfprmrg5spi3jbe8qx1k&st=viyrdlnr&raw=1",
    }

    if not download_from_dropbox(model_url, model_path):
        raise RuntimeError("Model download failed.")

    for name, url in tokenizer_files.items():
        path = os.path.join(tokenizer_dir, name)
        if not download_from_dropbox(url, path):
            raise RuntimeError(f"Failed to download {name}")

    model = SentimixtureNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer

# ------------------------------
# Feedback Logging Function
# ------------------------------
def get_gsheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_path = "/mnt/data/dauntless-loop-412713-2291c976a99e.json"
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    return gspread.authorize(creds)

def log_feedback_to_gsheet(tweet, prediction, confidence, user_feedback):
    try:
        client = get_gsheet_client()
        sheet = client.open("Sarcasm_Feedback").sheet1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [now, tweet, prediction, f"{confidence:.2%}", user_feedback]
        sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"❌ Failed to log feedback: {e}")
        return False

# ------------------------------
# (Insert rest of UI and detection logic here)
# Feedback Buttons Usage Example:
# if detect:
#     st.session_state.last_prediction = {
#         "text": text,
#         "label": label,
#         "confidence": confidence
#     }
#     ...
#     if st.button("👍 Yes, correct"):
#         log_feedback_to_gsheet(text, label, confidence, "Yes")
#     if st.button("👎 No, incorrect"):
#         log_feedback_to_gsheet(text, label, confidence, "No")
