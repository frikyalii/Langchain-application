import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Build the base serverless engine using a cloud-supported repository
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    temperature=0.1  # <-- Temperature belongs here, not down below!
)

# Pass the compliant serverless layer directly to ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Execute query 
result = model.invoke("What is the capital of Italy,India,France?")
print(result.content)

