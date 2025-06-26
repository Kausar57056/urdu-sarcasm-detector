import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, XLMRobertaModel
from huggingface_hub import hf_hub_download

# Define your model
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

# Streamlit UI setup
st.set_page_config(page_title="Urdu Sarcasm Detector", page_icon="😏", layout="centered")

st.markdown(
    """
    <style>
    .big-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    .small-subtitle {
        font-size: 18px;
        text-align: center;
        margin-bottom: 25px;
        color: #666;
    }
    .footer {
        margin-top: 50px;
        font-size: 13px;
        color: gray;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">😏 Urdu Sarcasm Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="small-subtitle">Enter an Urdu tweet below and find out if it\'s sarcastic or not!</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    try:
        with st.status("Loading resources...", expanded=True):
            st.write("🔄 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("kausar57056/urdu-sarcasm-detect")

            st.write("📁 Downloading model file...")
            model_path = hf_hub_download(
                repo_id="kausar57056/urdu-sarcasm-detect",
                filename="sentimixture_model.pt"
            )
            st.write(f"✅ Model file downloaded to: {model_path}")

            st.write("📦 Loading model weights...")
            model = SentimixtureNet()
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()

            st.write("✅ Model & Tokenizer are ready!")
            return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model or tokenizer:\n{e}")
        raise e

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Input box
text = st.text_area("✍️ Write your Urdu tweet here:", height=150, placeholder="مثال: واہ کیا ہی زبردست سروس ہے (sarcastic)")

# Prediction
if st.button("🔍 Detect Sarcasm"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some Urdu text.")
    else:
        with st.spinner("Analyzing... Please wait..."):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    pred = torch.argmax(logits, dim=1).item()
                    label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                    st.success(f"🎯 *Prediction*: {label}")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# Footer
st.markdown('<div class="footer">Built with ❤️ using PyTorch, Hugging Face, and Streamlit</div>', unsafe_allow_html=True)
