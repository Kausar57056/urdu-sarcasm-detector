import streamlit as st
import torch
import torch.nn as nn
import os
import requests
from transformers import AutoTokenizer, XLMRobertaModel

# ------------------------------
# SentimixtureNet Model Class
# ------------------------------
class SentimixtureNet(nn.Module):
    def __init__(self):
        super(SentimixtureNet, self).__init__()
        self.base = XLMRobertaModel.from_pretrained("tokenizer/")
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

    # Tokenizer files
    tokenizer_files = {
        "sentencepiece.bpe.model": "https://www.dropbox.com/scl/fi/zb8h3hpt2g7glopn2xhqx/sentencepiece.bpe.model?rlkey=eqwaym45sg419b5m67zxs0jfs&st=97sy243j&dl=0",
        "special_tokens_map.json": "https://www.dropbox.com/scl/fi/ehiwmv4wwiveto6lqleac/special_tokens_map.json?rlkey=q4gpcw7z9nwk97cmlgxrb52ky&st=zng6v2qx&dl=0",
        "tokenizer_config.json": "https://www.dropbox.com/scl/fi/4kv21xx93zqs6koarzvca/tokenizer_config.json?rlkey=5jdqzmfprmrg5spi3jbe8qx1k&st=viyrdlnr&dl=0"
    }

    # Download model
    if not download_from_dropbox(model_url, model_path):
        raise RuntimeError("Model download failed.")

    # Download tokenizer files
    for name, url in tokenizer_files.items():
        path = os.path.join(tokenizer_dir, name)
        if not download_from_dropbox(url, path):
            raise RuntimeError(f"Failed to download {name}")

    # Load model
    model = SentimixtureNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Urdu Sarcasm Detector")
st.title("😏 Urdu Sarcasm Detector")
st.write("Paste an Urdu tweet below to check if it's sarcastic or not.")

try:
    model, tokenizer = load_model_and_tokenizer()
    st.success("✅ Model & tokenizer loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model/tokenizer: {e}")
    st.stop()

# Input
text = st.text_area("✍️ Enter Urdu Tweet:")

if st.button("Detect Sarcasm"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = torch.softmax(logits, dim=1).squeeze()
                    pred = torch.argmax(probs).item()
                    confidence = probs[pred].item()
                    label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                    st.success(f"Prediction: *{label}\n\nConfidence: *{confidence:.2%}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
