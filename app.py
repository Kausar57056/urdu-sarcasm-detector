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
    import json
    import tempfile

    # Convert st.secrets["gsheets"] to a regular dict
    secrets_dict = dict(st.secrets["gsheets"])

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(secrets_dict, temp_file)
        temp_file.flush()
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            temp_file.name,
            scopes=[
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
        )
    return gspread.authorize(creds)


def log_feedback_to_gsheet(tweet, prediction, confidence, user_feedback):
    try:
        client = get_gsheet_client()
        sheet = client.open("Sarcasm_Feedback").sheet1  # Make sure this is the exact name of your sheet
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [now, tweet, prediction, f"{confidence:.2%}", user_feedback]
        sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"❌ Failed to log feedback: {e}")
        return False

# ------------------------------
# App Title & Subtitle (Centered, with Emoji)
# ------------------------------

st.markdown("""
    <h1 style='text-align: center; color: #002b5c; font-weight: 700; letter-spacing: 0.5px;'>
        Urdu Sarcasm Detector
    </h1>
    <p style='text-align: center; font-size: 17px; color: #444; margin-top: -10px;'>
        Identify sarcasm in Urdu tweets using advanced deep learning and XLM-RoBERTa
    </p>
""", unsafe_allow_html=True)


# ------------------------------
# Input Text + Detect Button
# ------------------------------

st.subheader("📝 Paste or type an Urdu tweet")

def set_example(example_text):
    st.session_state.input_text = example_text

text = st.text_area(
    " ",
    height=150,
    placeholder="مثال: واہ جی، بہت ہی بہترین سروس ہے، تین گھنٹے سے انتظار کر رہا ہوں۔",
    key="input_text"
)

# Examples
st.markdown("💡 **Examples:**")
examples = [
    "جیسے ھو ویسے رہو ویسے نظر آو۔ بس",
    "` مراد علی شاہ کے بھیس میں ڈی جی آئی ایس آئی تھے '' حامد میر",
    "مریم نواز کو انگلش نہیں آتی اور بلاول صاحبہ کو اردو نہیں آتی اور انکے سپورٹرز کو شرم نہیں آتی",
    "کامران خان صاحب آپ کیوں ذلالت کی چوٹی پر پہنچنا چاہ رہے ہیں"
]
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    cols[i].button(example, key=f"ex{i}", on_click=set_example, args=(example,))

# Centered detect button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    detect = st.button("🔎 Detect Sarcasm")

# Detect button centered
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin: auto;
        width: 200px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

detect = st.button("🔍 Detect Sarcasm")

# ------------------------------
# Load model and tokenizer
# ------------------------------
try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    st.error(f"❌ Failed to load model/tokenizer: {e}")
    st.stop()

# ------------------------------
# Save prediction result
# ------------------------------
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
                    bar_color = "#ff6666" if pred == 1 else "#66bb6a"

                    st.session_state.last_prediction = {
                        "text": text,
                        "label": label,
                        "confidence": confidence
                    }

                    st.markdown(f"""
                        <div class=\"result-box\">
                            <h4>{label}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            <p><strong>Tweet:</strong> {text}</p>
                        </div>
                        <style>
                            .confidence-bar {{
                                background-color: #ddd;
                                border-radius: 10px;
                                overflow: hidden;
                                height: 20px;
                                width: 100%;
                                margin-top: 10px;
                            }}
                            .confidence-fill {{
                                height: 100%;
                                width: {confidence * 100:.2f}%;
                                background-color: {bar_color};
                                border-radius: 10px;
                                transition: width 0.5s ease-in-out;
                            }}
                        </style>
                        <div class=\"confidence-bar\">
                            <div class=\"confidence-fill\"></div>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ------------------------------
# Feedback Section
# ------------------------------
st.markdown("### Did we get it right?")
col_yes, col_no = st.columns(2)

with col_yes:
    if st.button("👍 Yes, correct"):
        pred = st.session_state.get("last_prediction", {})
        if pred:
            if log_feedback_to_gsheet(pred["text"], pred["label"], pred["confidence"], "Yes"):
                st.success("Thanks for your feedback! 🙌")

with col_no:
    if st.button("👎 No, incorrect"):
        pred = st.session_state.get("last_prediction", {})
        if pred:
            feedback = st.text_input("Tell us what went wrong (optional):", key="feedback_input")
            if st.button("Submit Feedback"):
                if log_feedback_to_gsheet(pred["text"], pred["label"], pred["confidence"], feedback or "No"):
                    st.warning("Thanks! We'll use your feedback to improve. 💡")
