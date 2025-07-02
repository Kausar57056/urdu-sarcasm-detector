import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, XLMRobertaModel
from huggingface_hub import hf_hub_download

# -----------------------------
# Define SentimixtureNet model
# -----------------------------
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

# -----------------------------
# Load model & tokenizer
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    try:
        st.write("🔄 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("kausar57056/urdu-sarcasm-detect")

        st.write("📁 Downloading model file...")
        model_path = hf_hub_download(
            repo_id="kausar57056/urdu-sarcasm-detect",
            filename="sentimixture_model.pt"
        )
        st.write(f"✅ Model file downloaded to: {model_path}")

        st.write("📦 Initializing model architecture...")
        model = SentimixtureNet()

        st.write("📥 Loading model weights...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        st.success("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model or tokenizer:\n\n{e}")
        raise e

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Urdu Sarcasm Detector", layout="centered")
st.title("😏 Urdu Sarcasm Detection")
st.write("Enter an Urdu tweet to detect if it's sarcastic or not.")

# Load model and tokenizer once
model, tokenizer = load_model_and_tokenizer()

# Input
text = st.text_area("✍️ Write your Urdu tweet here:")

# Prediction
if st.button("🔍 Detect Sarcasm"):
    if not text.strip():
        st.warning("⚠️ Please enter some Urdu text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    pred = torch.argmax(logits, dim=1).item()
                    label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                    st.success(f"*Prediction:* {label}")
            except Exception as e:
                st.error(f"❌ Error during prediction:\n\n{e}")
