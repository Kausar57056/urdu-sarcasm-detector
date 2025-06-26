import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, XLMRobertaModel
from huggingface_hub import hf_hub_download

# 🚀 Model Architecture
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

# 🌟 UI Setup
st.set_page_config(page_title="Urdu Sarcasm Detector", layout="centered")
st.markdown("<h1 style='text-align: center;'>😏 Urdu Sarcasm Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: grey;'>Enter an Urdu tweet below to find out if it's <strong>sarcastic</strong> or not!</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# 📦 Load model & tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        with st.spinner("🔄 Downloading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained("kausar57056/urdu-sarcasm-detect")

        with st.spinner("📁 Downloading model weights..."):
            model_path = hf_hub_download(
                repo_id="kausar57056/urdu-sarcasm-detect",
                filename="sentimixture_model.pt"
            )

        with st.spinner("📦 Loading model..."):
            model = SentimixtureNet()
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()

        st.success("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer

    except Exception as e:
        st.error(f"❌ Error loading model/tokenizer:\n\n`{e}`")
        raise e

# 🔧 Load resources
model, tokenizer = load_model_and_tokenizer()

# ✍️ User Input
text = st.text_area("📝 Write your Urdu tweet below:", height=150, placeholder="مثال: آج موسم بہت زبردست ہے، بارش بھی ہو رہی ہے 😒")

# 🔍 Predict
if st.button("🔎 Detect Sarcasm"):
    if text.strip() == "":
        st.warning("⚠️ Please enter a valid Urdu sentence.")
    else:
        with st.spinner("🔍 Analyzing..."):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    pred = torch.argmax(logits, dim=1).item()
                    label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                    st.success(f"*Prediction:* {label}")
            except Exception as e:
                st.error(f"❌ Error during prediction:\n\n`{e}`")

# 🔚 Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px;'>Built with ❤️ using Streamlit and Hugging Face Transformers</p>", unsafe_allow_html=True)
