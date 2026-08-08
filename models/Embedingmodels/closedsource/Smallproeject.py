# Import tool to create embeddings using OpenAI
from langchain_openai import OpenAIEmbeddings

# Import tool to load secret keys from .env file
from dotenv import load_dotenv

# Import tool to calculate similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity

# Import numpy (helper library for numbers/arrays)
import numpy as np

# Load the .env file so we can use our OpenAI API key
load_dotenv()

# Set up the embedding model (each text becomes a 300-number vector)
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# Step 0: our list of documents (info about cricketers)
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# The question we want to find the best matching document for
query = 'tell me about virat kohli'

# Step 1: calculate embeddings for all documents (turn each sentence into a vector)
doc_embeddings = embedding.embed_documents(documents)

# Step 2: calculate embedding for the query (turn the question into a vector)
query_embedding = embedding.embed_query(query)

# Step 3: find cosine similarity between query and each document
# This tells us how "close in meaning" the query is to each document
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Sort scores and pick the document with the highest similarity (best match)
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Print the query
print(query)

# Print the best matching document
print(documents[index])

# Print how similar it was (closer to 1 = more similar)
print("similarity score is:", score)


## ALSSO SEMATIC SERCH I ALSSP DOEN JHEREE
