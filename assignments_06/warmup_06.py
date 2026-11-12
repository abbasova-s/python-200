
from dotenv import load_dotenv
import os

# Load environment variables from the .env file.
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts --- 
# Concepts Question 1

# Scenario A: RAG.
# Since the company has hundreds of internal policy PDFs that are updated regularly, RAG would be the best choice. It retrieves relevant information from the documents and adds it to the LLM's context. When quarterly updates arrive, the document index can be updated without retraining the model.

# Scenario B: Fine-tuning.
# Since the company has 3,000 examples of its preferred writing style, fine-tuning would be a good choice. These examples can help the model learn the specific brand voice and reproduce it more consistently.

# Scenario C: Prompt engineering.
# Since the analyst only needs to work with one two-page report, she can include the report directly in the prompt and ask questions about it. There is no need to build a separate RAG pipeline for a one-time task.



# Concepts Question 2
# A confidently wrong answer is more harmful because users may trust it and make important decisions based on incorrect information instead of questioning or verifying the response. For example, if a user describes several symptoms and an AI confidently tells them they have cancer without sufficient evidence, it could cause unnecessary panic, emotional distress, and potentially influence their medical decisions.

# The tone of the response also affects how much users trust the model. A confident and authoritative tone can make incorrect information sound reliable and convincing. On the other hand, when a model admits uncertainty, users are more likely to question the answer and seek confirmation from reliable sources or qualified professionals.


# Concepts Question 3
# 1. Extract text from source documents
# Extract readable text from documents such as PDFs.

# 2. Split text into chunks
# Divide the extracted text into smaller pieces so they can be processed and retrieved more efficiently.

# 3. Convert text chunks into embeddings
# Transform each text chunk into a numerical vector that represents its semantic meaning.

# 4. Receive the user's query
# Accept the question or request submitted by the user.

# 5. Embed the user's query
# Convert the user's question into a numerical vector using the same embedding model.

# 6. Retrieve the most relevant chunks
# Compare the query embedding with document embeddings to find the chunks that are most semantically similar.

# 7. Inject retrieved chunks into the prompt
# Add the retrieved information to the LLM prompt as context.

# 8. Generate a response from the LLM
# Use the user's query and retrieved context to generate a relevant response.



# --- Keyword RAG --- 
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Question 1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


result = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {result[0][0]}")

# The function selects loyalty.txt because the query contains "your", which is not included in the stopwords list. This creates a tie between loyalty.txt, hiring.txt, and hours.txt (each has one matching keyword). Reverse tuple sorting selects loyalty.txt alphabetically among the tied results.



# Keyword Question 2
query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {result[0][0]}")

# The function did not find a matching document and returned "None found" because none of the query keywords appear in the documents. However, menu.txt is the most relevant document because the question is about caffeine-free drink options. Keyword retrieval misses this connection because it only checks for exact word overlap and does not understand the meaning or context of the query. Semantic retrieval using embeddings would do better because it compares the meaning of the query with the meaning of the document, rather than relying only on matching words.


# Keyword Question 3
# Prediction: I expected loyalty.txt to be selected because the question is about signing up for rewards, which is related to the loyalty program described in that document.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {result[0][0]}")

# My prediction was incorrect. The function returned "None found" because none of the query keywords appear exactly in loyalty.txt. Although "rewards" is related to a "loyalty program" and "sign up" is similar to "join", keyword retrieval cannot recognize synonyms or understand the meaning of related phrases.



# --- Semantic RAG Concepts --- 
# Semantic Question 1

# What is a vector embedding? (1-2 sentences)
# A vector embedding is a list of numbers that represents the meaning of a piece of text. Texts with similar meanings tend to have embeddings that are close to each other in the vector space.

# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30.
# The chunk with a score of 0.85 is more relevant because its meaning is more similar to the query. A higher cosine similarity score means the two text embeddings point in a more similar direction, indicating a stronger semantic relationship.

# 3. Why can semantic search find a relevant chunk without exact keywords?
# Semantic search compares the meanings represented by text embeddings rather than looking only for matching words. This allows it to recognize synonyms, related phrases, and different ways of expressing the same idea, even when the query and the text have no exact words in common.



# Semantic Question 2
#| Feature                    | Keyword RAG                       | Semantic RAG                 |
#|----------------------------|-----------------------------------|------------------------------|
#| What is compared?          | Exact word overlap                | text embeddings              |
#| What is retrieved?         | Full document                     | Most relevant text chunks    |
#| Can it handle synonyms?    | No                                | yes                          |
#| Storage format             | Plain text dictionary             | Vector store with embeddings |
#| Relevance score            | Number of overlapping keywords    | cosine similarity            |



# --- LlamaIndex --- 
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Configure the embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Configure the language model
Settings.llm = OpenAI(model="gpt-4.1-mini")

# Load the BrightLeaf Solar PDF documents
docs = SimpleDirectoryReader("lessons/06_AI_augmentation/resources/brightleaf_pdfs").load_data()

# Build an in-memory vector index
index = VectorStoreIndex.from_documents(docs)

# LlamaIndex Question 1
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f'\nQ: {q}')
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-"*30)


# Observations:

# Query 1: Employee benefits
# The first retrieved chunk (Introduction) was highly relevant and contained information about employee benefits. However, the EcoVolt partnership and Overview chunks were less relevant and somewhat unexpected. The model provided a confident and detailed response, including specific benefits and financial figures, without noticeable hedging.

# Query 2: Security policies
# The first retrieved chunk (Network and Data Security) was directly relevant and contained the security policies. However, the Introduction and EcoVolt partnership chunks were unexpected because they were not directly related to the question. The model provided a confident and specific response, including technical details and security procedures, without noticeable hedging.

# Overall, semantic retrieval found relevant information for both questions, even though some retrieved chunks were less relevant. This shows that retrieving the top three chunks does not guarantee that all of them will be useful to the final answer.


# LlamaIndex Question 2
# with similarity_top_k=1
query_engine = index.as_query_engine(similarity_top_k=1)

questions = [
    "What employee benefits does BrightLeaf offer?",
]

for q in questions:
    print(f'\nQ: {q}')
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-"*30)


# with similarity_top_k=5
query_engine = index.as_query_engine(similarity_top_k=5)

questions = [
    "What employee benefits does BrightLeaf offer?",
]

for q in questions:
    print(f'\nQ: {q}')
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-"*30)

# Observations:

# With similarity_top_k=1, the model retrieved only the Introduction chunk, which contained the relevant employee benefits information. It produced a detailed and specific answer based on that context.

# With similarity_top_k=5, the model retrieved five chunks, but only the first one was directly relevant. The other four contained mostly unrelated information about the company, its partnership, financial performance, and solar products.

# The response remained detailed and relevant, although some wording and details changed slightly between the two runs. Retrieving more context did not significantly improve the answer in this example.

# More retrieved context is not always better because irrelevant information can distract the model or introduce unnecessary details. The goal is to retrieve the most relevant information, not simply the largest number of chunks.



# LlamaIndex Question 3
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "Where do the BrightLeaf workers go for a happy hour?",
]

for q in questions:
    print(f'\nQ: {q}')
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()}...")
        print("-"*30)

# I expected the pipeline might struggle with this question because the documents may not contain information about employee happy hours. I thought the model might try to generate an answer using unrelated information retrieved from the documents.

# However, the model admitted that there was not enough information in the provided context to answer the question instead of making up a location or an event.

# To handle this type of query better, I would improve the system by adding instructions to the prompt telling the model to acknowledge when the retrieved information does not contain an answer. I would also introduce a minimum similarity threshold so that the system can recognize when retrieved chunks are not relevant enough to answer the question.


# LlamaIndex Question 4
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# First question
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Evaluation: " + str(relevancy_result.score))


# Second question
q = "What is employees of BrightLeaf favorite color?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Evaluation: " + str(relevancy_result.score))

print("Response:", response)

print("Faithfulness feedback:", faithfulness_result.feedback)
print("Relevancy feedback:", relevancy_result.feedback)


# Observations:

# Both queries received a faithfulness score of 1.0 and a relevancy score of 1.0.

# 1. A faithfulness score of 1.0 means the evaluator judged the response to be supported by the retrieved context. A score of 0.0 would indicate that the response contained information that was not sufficiently supported by the context.

# 2. Faithfulness and relevancy measure different things. Faithfulness checks whether the answer is supported by the retrieved information, while relevancy checks whether the answer addresses the user's question.

# 3. The scores did not change between the two queries. For the first question, the model provided information about employee benefits. For the second question, it acknowledged that the documents did not contain information about employees' favorite color instead of inventing an answer. The evaluator considered both responses faithful and relevant.

# 4. LLM-as-a-judge is useful because it can evaluate answers written in different ways by comparing them with the original question and retrieved context. Exact string matching would be too limited for evaluating natural language responses. However, an LLM judge is not perfect and its scores should be treated as estimates rather than completely objective measurements.




