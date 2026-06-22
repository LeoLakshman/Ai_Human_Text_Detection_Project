import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils.text_features import extract_linguistic_matrix, simple_clean_text, LINGUISTIC_FEATURE_NAMES

sns.set_style("whitegrid")

df = pd.read_excel('/mnt/user-data/uploads/train_data_with_labels.xlsx')
df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
df['label'] = df['label'].astype(int)
df['text_clean'] = df['text'].apply(simple_clean_text)
df['word_count'] = df['text_clean'].str.split().apply(len)
df['char_count'] = df['text_clean'].str.len()

print("Shape:", df.shape)
print(df['label'].value_counts())
print(df.isna().sum())
print(df.duplicated(subset=['text']).sum(), "exact duplicate texts")

# --- Plot 1: class balance ---
fig, ax = plt.subplots(figsize=(5, 4))
counts = df['label'].value_counts().sort_index()
bars = ax.bar(['Human (0)', 'AI (1)'], counts.values, color=['#4C72B0', '#DD8452'])
ax.set_title('Class Balance')
ax.set_ylabel('Number of documents')
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 30, str(v), ha='center')
plt.tight_layout()
plt.savefig('eda_class_balance.png', dpi=130)
plt.close()

# --- Plot 2: word count distribution by label ---
fig, ax = plt.subplots(figsize=(7, 4))
for lbl, name, color in [(0, 'Human', '#4C72B0'), (1, 'AI', '#DD8452')]:
    sns.kdeplot(df.loc[df.label == lbl, 'word_count'], label=name, fill=True, alpha=0.3, ax=ax, color=color)
ax.set_title('Word Count Distribution by Label')
ax.set_xlabel('Word count')
ax.legend()
plt.tight_layout()
plt.savefig('eda_word_count_dist.png', dpi=130)
plt.close()

print(df.groupby('label')['word_count'].describe())

# --- Linguistic features for EDA visualization ---
ling = extract_linguistic_matrix(df['text_clean'].tolist())
ling_df = pd.DataFrame(ling, columns=LINGUISTIC_FEATURE_NAMES)
ling_df['label'] = df['label'].values
ling_df.to_csv('linguistic_features_full.csv', index=False)
print(ling_df.groupby('label').mean().T)

# --- Plot 3: a few linguistic features by label ---
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
feats_to_plot = ['type_token_ratio', 'flesch_reading_ease', 'avg_sentence_length', 'punctuation_density']
for ax, feat in zip(axes.flat, feats_to_plot):
    sns.boxplot(data=ling_df, x='label', y=feat, ax=ax, palette=['#4C72B0', '#DD8452'])
    ax.set_xticklabels(['Human', 'AI'])
    ax.set_title(feat)
plt.tight_layout()
plt.savefig('eda_linguistic_boxplots.png', dpi=130)
plt.close()

# --- Train/test split (stratified) ---
train_df, test_df = train_test_split(
    df[['text', 'text_clean', 'label']], test_size=0.2, random_state=42, stratify=df['label']
)
train_df.to_csv('data/training_data/train.csv', index=False)
test_df.to_csv('data/test_data/test.csv', index=False)
print("Train:", train_df.shape, "Test:", test_df.shape)
print("Train label balance:\n", train_df['label'].value_counts())
print("Test label balance:\n", test_df['label'].value_counts())
