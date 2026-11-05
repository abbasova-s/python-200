# --- Completions API --- 
# API Q1

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)


# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    print(f"\nTemperature: {temperature}")
    print("Response:", response.choices[0].message.content)

# o temperature gives the most consistent, reproducible output. The higher thetemperature, the more creative the response is. I would use temperature o for acquiring consistent response


# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"Completion {i}: {choice.message.content}")


#API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Explain how the universe was created."
    }],
    max_tokens=15
)

print("Response:", response.choices[0].message.content)

# It gave a short narrow response and in real world measures such as max_tokens would help save money for a company, since it controls the response length; token usage.


# --- System Messages and Personas --- 
# System Q1
# First personality
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Patient Tutor:")
print(response.choices[0].message.content)


# Second personality
messages = [
    {
        "role": "system",
        "content": "You are an aggresive pirate. You explain everything using pirate language and get frastruated that others are too stupid to understand simple things."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Pirate:")
print(response.choices[0].message.content)

# The content of both responses was very similar, however the tone and style were different depending on a set personality.


# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# The model knows the name Jordan because the previous messages were included in this call. While API itself does not remember past calls/ stateless, it's possible to provide history of messages each time


# --- Prompt Engineering --- 
# Prompt Q1 — Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Classify the sentiment of this review as positive, negative, or mixed. Return only the classification. Review: {review}"
            }
        ]
    )

    print(f"Zero-shot review {i}: {response.choices[0].message.content}")


# Prompt Q2 — One-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Classify the sentiment of each review as positive, negative, or mixed.
                
                Example:
                Review: "Fast shipping but the item arrived damaged."
                Sentiment: mixed

                Review: "{review}"
                Sentiment:"""
            }
        ]
    )


    print(f"One-shot review {i}: {response.choices[0].message.content}")

# Adding one example did not change the classifications, but it helped produce consistent lowercase labels matching the example's format.


# Prompt Q3 — Few-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Classify the sentiment of each review as positive, negative, or mixed. Return only the lowercase sentiment label.

Example 1:
Review: "The product is easy to use and works perfectly."
Sentiment: positive

Example 2:
Review: "The product is unreliable and customer service is terrible."
Sentiment: negative

Example 3:
Review: "The delivery was fast, but the packaging was damaged."
Sentiment: mixed

Review: "{review}"
Sentiment:"""
            }
        ]
    )

    print(f"Few-shot review {i}: {response.choices[0].message.content}")

# Zero-shot is useful for simple tasks when instructions are clear. One-shot is useful when one example can clarify the expected format or behavior. Few-shot is useful when the task has multiple categories or requires more guidance and consistency.




# Prompt Q4 — Chain of Thought
prompt = """Show your step-by-step reasoning before giving the final answer on its own line labelled: Final answer: <value>

Problem:
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary?"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)

# Asking the model to break a problem into steps helps it organize the calculations, follow the sequence of events, and reduce arithmetic mistakes. 


# Prompt Q5 — Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""Analyze the review below and return the result only as valid JSON 
Keys: sentiment, confidence (a float from 0 to 1), and reason (one sentence)

Review: {review}"""



response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

raw_response = response.choices[0].message.content
print("Raw response: ", raw_response)

try:
    result = json.loads(raw_response)
    print("Parsed sentiment: ", result["sentiment"])
    print("Confidence: ", result["confidence"])
    print("Reason: ", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")
    print("Raw response for debugging:", raw_response)


# Prompt Q6 — Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

print("Instructions result:")
print(response.choices[0].message.content)


# Second prompt
user_text = "The mountains were covered in snow. The morning sun created beautiful reflections across the frozen lake."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

print("\nRegular prose result:")
print(response.choices[0].message.content)

# Delimiters clearly separate user-provided text from instructions, helping prevent the model from confusing the text to be analyzed with instructions it should follow.



# --- Local Models with Ollama --- 
# Ollama Q1

# Ollama response:
"""
A large language model is an artificial intelligence system that 
can understand and generate human-like text, trained on vast 
datasets and using complex algorithms to comprehend and produce 
diverse content. It allows for tasks like writing, answering 
questions, and even creative writing by analyzing massive amounts 
of text.
"""

# OpenAI response:
prompt = """Explain what a large language model is in two sentences."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)

# OpenAI response: A large language model is an advanced artificial intelligence system designed to understand and generate human language by analyzing vast amounts of textual data. It uses deep learning techniques, particularly neural networks, to predict the next words in a sequence, enabling it to generate coherent and contextually relevant text.


# Both responses are similar with a minor difference: Ollama response is simpler and provides examples, while OpenAI explains all in more details. Both followed the prompt. Advantage of running a model locally: gives more control over privacy, disadvantage: require computer memory and processing power to run.

