# Step 1: Setup
from dotenv import load_dotenv
import os
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pathlib import Path


# Load environment variables from the .env file.
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


docs_dir = Path("lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# Step 2: Load the Documents
docs = SimpleDirectoryReader(docs_dir).load_data()

print(len(docs), " documents were loaded")
for doc in docs:
    print(doc.metadata["file_name"])

# Step 3: Build the Index and Query Engine
# Configure the embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Configure the language model
Settings.llm = OpenAI(model="gpt-4.1-mini")

# Build an in-memory vector index
index = VectorStoreIndex.from_documents(docs)

# LlamaIndex Question 1
query_engine = index.as_query_engine(similarity_top_k=3)

if query_engine:
    print("Index built successfully. Ready to answer questions.")

# Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f'\nQ: {q}')
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Document Name: {node_with_score.node.metadata["file_name"]}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-"*30)

# All five answers sounded confident and appeared accurate based on the provided documents. However, I noticed that the most relevant document was not always ranked first. For example, the weekend-hours question retrieved our_story.txt before faq.txt, even though the FAQ contained the specific hours. The answer was still correct, but this showed me that a model can sometimes produce a good answer even when the retrieval ranking is not ideal.


# Step 5: Find a Failure
q = "Who is being discriminated at Groundwork locations in Asheville?"
response = query_engine.query(q)

print(f'\nQ: {q}')
print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Document Name: {node_with_score.node.metadata["file_name"]}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
    print("-"*30)

# I asked who is being discriminated against at Groundwork locations in Asheville. I expected this question to be difficult because the documents describe the business, but may not contain specific information about discrimination incidents.

# The model answered that no one is being discriminated against and referred to the company's commitment to creating an inclusive space. However, a company's values or commitment to inclusion do not prove that discrimination has never occurred. The model made a broad conclusion without sufficient evidence.

# The model still sounded confident even though its conclusion was not necessarily supported by the retrieved information. This shows why users should not automatically trust confident AI-generated responses.

# To improve the system, I would add stricter instructions telling the model to answer only when the retrieved documents provide sufficient evidence. If the information is missing, it should explicitly say that it cannot determine the answer from the available documents.

# Step 6: Reflection
# 1. LlamaIndex allowed me to build a RAG pipeline with just a few lines of code. Implementing chunking, embeddings, and retrieval manually would require significantly more code. This shows how frameworks simplify development and save time.

# 2. A law firm could use RAG to search through hundreds of client documents and quickly retrieve relevant information, saving lawyers time during case preparation.

# 3. RAG cannot completely prevent hallucinations. Even with relevant documents, an LLM can misunderstand information or make unsupported conclusions and present them confidently.







