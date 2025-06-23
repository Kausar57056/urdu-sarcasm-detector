import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^؀-ۿ ]', '', text)
    return text.strip()

def load_and_prepare_dataset(filename):
    df = pd.read_csv(filename)
    df = df[['urdu_text', 'is_sarcastic']]
    df.dropna(inplace=True)
    df['urdu_text'] = df['urdu_text'].apply(clean_text)
    df['is_sarcastic'] = df['is_sarcastic'].astype(int)
    return train_test_split(df['urdu_text'], df['is_sarcastic'], test_size=0.2, stratify=df['is_sarcastic'])