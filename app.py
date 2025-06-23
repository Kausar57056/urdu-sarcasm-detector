import streamlit as st
from transformers import XLMRobertaTokenizer
import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from huggingface_hub import hf_hub_download

# Define your custom model
class SentimixtureNet(nn.Module):
    def init(self):
        super(SentimixtureNet, self).init()
        self.base = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.routing = nn.Linear(768, 768)
        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        routed = torch.r…
[11:24 pm, 23/06/2025] K✨: import streamlit as st
from transformers import XLMRobertaTokenizer
import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from huggingface_hub import hf_hub_download

# Define your custom model
class SentimixtureNet(nn.Module):
    def __init__(self):  # ✅ Corrected constructor
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

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimixtureNet().to(device)

# ✅ Download model weights from Hugging Face
model_path = hf_hub_download(repo_id="kausar57056/urdu-sarcasm-detect", filename="sentimixture_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Load tokenizer from Hugging Face (make sure tokenizer files were uploaded there)
tokenizer = XLMRobertaTokenizer.from_pretrained("kausar57056/urdu-sarcasm-detect")

# Streamlit UI
st.set_page_config(page_title="Urdu Sarcasm Detector", layout="centered")
st.title("🧠 Urdu Sarcasm Detection")
st.write("Enter an Urdu tweet to detect if it's sarcastic or not.")

text = st.text_area("✍️ Write your Urdu tweet here:")

if st.button("🔍 Detect Sarcasm"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some Urdu text.")
    else:
        with st.spinner("Analyzing..."):
            enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc)
                pred = torch.argmax(logits, dim=1).item()
                label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                st.success(f"Prediction: {label}")
