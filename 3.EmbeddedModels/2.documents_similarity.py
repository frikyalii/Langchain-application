from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=300)

documents = [
    "Virat Kohli is a cricketer known for aggresive batting and leadership qualities ",
    "Rohit Sharma is a cricketer known for his calm and composed batting style",
    "MS Dhoni is a cricketer known for his leadership qualities and his ability to win matches single-handedly",
    "Sachin Tendulkar is a cricketer known for his batting skills and his ability to win matches single-handedly",
    "Yuvraj Singh is a cricketer known for his batting skills and his ability to win matches single-handedly",
]

query = "Tell me about MS Dhoni"

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print(f"Similarity score: {score}")

