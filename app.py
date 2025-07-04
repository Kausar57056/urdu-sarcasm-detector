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
# Enhanced Streamlit UI - V2
# ------------------------------
st.set_page_config(page_title="Urdu Sarcasm Detector", page_icon="😏", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
        body {
            background-color: #f4f6f8;
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        .result-box {
            background: linear-gradient(to right, #ffffff, #f0f0f0);
            border-left: 6px solid #6c63ff;
            padding: 1em;
            border-radius: 8px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .example {
            color: #666;
            cursor: pointer;
        }
        .example:hover {
            color: #000;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h1 style='text-align: center;'>😏 Urdu Sarcasm Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🔍 Detect sarcasm in Urdu tweets using deep learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Input Section
st.subheader("📝 Paste or type an Urdu tweet")

def set_example(example_text):
    st.session_state.input_text = example_text

# Text area bound to session_state['input_text']
text = st.text_area(
    " ",
    height=150,
    placeholder="مثال: واہ جی، بہت ہی بہترین سروس ہے، تین گھنٹے سے انتظار کر رہا ہوں۔",
    key="input_text"
)

# Examples
st.markdown("💡 **Examples:**")
examples = [
    "کمال ہے، بارش میں بھی بجلی نہیں گئی، حیرت ہے۔",
    "`` مراد علی شاہ کے بھیس میں ڈی جی آئی ایس آئی تھے '' حامد میر",
    "کتنی اچھی نیند آئی آج، صرف تین بار الارم بجا۔",
    "اتنی اچھی فلم تھی کہ سارا ہال سو گیا۔"
]
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    cols[i].button(example, key=f"ex{i}", on_click=set_example, args=(example,))


# Centered Detect Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    detect = st.button("🔎 Detect Sarcasm")

# Load model
try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    st.error(f"❌ Failed to load model/tokenizer: {e}")
    st.stop()

# Prediction Logic
if detect:
    if not text.strip():
        st.warning("⚠️ Please enter or select a tweet.")
    else:
        with st.spinner("Analyzing for sarcasm..."):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = torch.softmax(logits, dim=1).squeeze()
                    pred = torch.argmax(probs).item()
                    confidence = probs[pred].item()
                    label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
                    color = "#ffcccc" if pred == 1 else "#d4edda"

                    st.markdown(f"""
                        <div class="result-box">
                            <h4>{label}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            <p><strong>Tweet:</strong> {text}</p>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
