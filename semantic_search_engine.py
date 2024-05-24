from datasets import load_dataset
from sentence_transformers import SentenceTransformer

dataset = load_dataset('multi_news', split="test")
df = dataset.to_pandas().sample(2000, random_state=42)
df.iloc[0]['summary']

model = SentenceTransformer("all-MiniLM-L6-V2")

passage_embeddings = model.encode(df['summary'].to_list(), show_progress_bar=True)

passage_embeddings = list(passage_embeddings)