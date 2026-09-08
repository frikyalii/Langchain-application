from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text = "Hello, how are you?"

embedding = embeddings.embed_query(text)
print(embedding)