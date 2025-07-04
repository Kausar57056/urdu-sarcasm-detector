import streamlit as st
import torch
import torch.nn as nn
import requests
import os
from transformers import AutoTokenizer, XLMRobertaModel
import torch.nn.functional as F

# ------------------------------
# Model Definition
# ------------------------------
class SentimixtureNet(nn.Module):
    def _init_(self):
        super(SentimixtureNet, self)._init_()
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
# Load Model & Tokenizer from Dropbox
# ------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    dropbox_url = (
        "https://www.dropbox.com/scl/fi/6p5l4kmjwglsx4bzp7u9t/sentimixture_model.pt"
        "?rlkey=qyii1kgniduebx4qyjagim0xk&st=pmmh1dut&dl=1"
    )
    model_path = "sentimixture_model.pt"

    with st.spinner("📦 Downloading and loading model..."):
        try:
            # Download model from Dropbox if not already present
            if not os.path.exists(model_path):
                response = requests.get(dropbox_url, stream=True)
                total_size = int(response.headers.get("Content-Length", 0))
                content_type = response.headers.get("Content-Type", "")

                if "html" in content_type:
                    raise RuntimeError("Dropbox returned HTML instead of the model. Check your link.")

                with open(model_path, "wb") as f:
                    downloaded = 0
                    for chunk in response.iter_content(1024 * 1024):  # 1MB
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = downloaded * 100 / total_size
                            st.write(f"⬇️ Downloaded {downloaded // 1024*2}MB / {total_size // 1024*2}MB ({percent:.1f}%)")

            if os.path.getsize(model_path) < 100_000_000:  # Ensure file is >100MB
                raise RuntimeError("Downloaded model is too small. Likely incomplete or corrupted.")

            # Load model
            model = SentimixtureNet()
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

            st.success("✅ Model and tokenizer loaded.")
            return model, tokenizer

        except Exception as e:
            st.error(f"❌ Failed to load model/tokenizer: {e}")
            raise e

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Urdu Sarcasm Detector", layout="centered")
st.title("😏 Urdu Sarcasm Detection")
st.write("Enter an Urdu tweet below to check if it's sarcastic or not.")

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# User input
text = st.text_area("✍️ Write your Urdu tweet here:")

# Detect button
if st.button("🔍 Detect Sarcasm"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some Urdu text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = F.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred].item()

                label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                st.success(f"*Prediction:* {label}")
                st.info(f"*Confidence:* {confidence * 100:.2f}%")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
