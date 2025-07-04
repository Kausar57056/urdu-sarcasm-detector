import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
from transformers import AutoTokenizer, XLMRobertaModel

# ------------------------------
# Custom Model Definition
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
# Model + Tokenizer Loader
# ------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    dropbox_url = "https://www.dropbox.com/scl/fi/6p5l4kmjwglsx4bzp7u9t/sentimixture_model.pt?rlkey=qyii1kgniduebx4qyjagim0xk&st=pmmh1dut&dl=1"
    model_path = "sentimixture_model.pt"

    with st.spinner("📦 Downloading and loading model..."):
        try:
            if not os.path.exists(model_path):
                response = requests.get(dropbox_url)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                else:
                    raise RuntimeError(f"Failed to download model from Dropbox. Status code: {response.status_code}")

            model = SentimixtureNet()
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

            st.success("✅ Model & tokenizer loaded successfully.")
            return model, tokenizer

        except Exception as e:
            st.error(f"❌ Failed to load model/tokenizer: {e}")
            raise e

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Urdu Sarcasm Detector", layout="centered")
st.title("😏 Urdu Sarcasm Detection")
st.write("Enter an Urdu tweet to detect if it's sarcastic or not.")

model, tokenizer = load_model_and_tokenizer()

text = st.text_area("✍️ Write your Urdu tweet here:")

if st.button("🔍 Detect Sarcasm"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some Urdu text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = F.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred].item() * 100
                    label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                    st.success(f"Prediction: {label} ({confidence:.2f}% confidence)")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
