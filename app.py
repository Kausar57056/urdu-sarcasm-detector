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

    # Tokenizer files
    tokenizer_files = {
    "sentencepiece.bpe.model": "https://www.dl.dropboxusercontent.com/scl/fi/zb8h3hpt2g7glopn2xhqx/sentencepiece.bpe.model?rlkey=eqwaym45sg419b5m67zxs0jfs&st=97sy243j&raw=1",
    "special_tokens_map.json": "https://www.dl.dropboxusercontent.com/scl/fi/ehiwmv4wwiveto6lqleac/special_tokens_map.json?rlkey=q4gpcw7z9nwk97cmlgxrb52ky&st=zng6v2qx&raw=1",
    "tokenizer_config.json": "https://www.dl.dropboxusercontent.com/scl/fi/4kv21xx93zqs6koarzvca/tokenizer_config.json?rlkey=5jdqzmfprmrg5spi3jbe8qx1k&st=viyrdlnr&raw=1",
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
# Enhanced Streamlit UI
# ------------------------------
st.set_page_config(page_title="Urdu Sarcasm Detector", page_icon="😏", layout="centered")
st.markdown(
    """
    <style>
        .main { background-color: #f9f9f9; }
        .stTextArea textarea {
            font-size: 16px;
        }
        .result-box {
            background-color: #f0f2f6;
            padding: 1.2em;
            border-radius: 10px;
            border: 1px solid #d3d3d3;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("😏 Urdu Sarcasm Detector")
st.caption("Detect sarcasm in Urdu tweets using a deep learning model trained on XLM-Roberta.")

# Input section
with st.container():
    st.subheader("📝 Paste an Urdu tweet:")
    text = st.text_area("Enter text here:", height=150, placeholder="مثال:انساں کو تھکا دیتا ہے سوچوں کا سفر بھی۔")

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    detect = st.button("🔍 Detect Sarcasm")

# Load model and tokenizer
try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    st.error(f"❌ Failed to load model/tokenizer: {e}")
    st.stop()

# Prediction logic
if detect:
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Analyzing for sarcasm..."):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = torch.softmax(logits, dim=1).squeeze()
                    pred = torch.argmax(probs).item()
                    confidence = probs[pred].item()
                    label = "😏 **Sarcastic**" if pred == 1 else "🙂 **Not Sarcastic**"
                    color = "#f8d7da" if pred == 1 else "#d4edda"
                    emoji = "😏" if pred == 1 else "🙂"

                    st.markdown(
                        f"""
                        <div class="result-box" style="background-color: {color};">
                            <h4>{emoji} Prediction: {label}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
