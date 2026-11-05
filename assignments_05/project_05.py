
# Task 1: Setup and System Prompt

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content
    print("Finish reason:", response.choices[0].finish_reason)
    print("Usage:", response.usage)


system_prompt = """
You are one of the top job aaplication coaches with 20+ years of experience. You are best at helping career changers to translate their experience from a previous field into language that resonates in a new one. 

Your role is to:
- Rewrite resume bullet points to highlight relevant skills and achievements.
- Translate experience from previous industries into language that
  resonates with employers in their target industry.
- Draft tailored cover letters based on the user's actual experience
  and the job description.
- Answer follow-up questions about resumes, cover letters, and job
  applications.

Behavioral constraints:
- Stay focused on job application materials
- Always remind the user to review and edit its output before submitting anywhere
- Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgment
- Ask clarifying questions if important information is missing
- Do not invent user's experience, skills or other qualifications
- Use clear, professional, and engaging language. Keep responses concise, informative, and measurable whenever possible.
"""

# I added instructions to not invent user's experience, skills or qualifications, so it accurately represents what a user has to offer in a new field


# Task 2: Bullet Point Rewriter
import json

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original. Do not add achievements, outcomes, metrics, or details that are not explicitly supported by the original bullet.

    - Preserve the exact meaning and factual scope of each original bullet.
    - Do not upgrade skill levels (for example, "know" to "proficient").
    - Do not invent responsibilities, achievements, tools, or results.
    - If a bullet is too vague to improve accurately, explain what additional information is needed.
    - If a bullet is unrelated to the target role, point that out instead of inventing relevance.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Respond ONLY with valid JSON, no other text.
    Do not use Markdown formatting or code fences.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]
    # call get_completion(), parse the JSON, and return the result
    response = get_completion(messages)
    results = json.loads(response)

    for result in results:
        print(f"Original: {result['original']}")
        print(f"Improved: {result['improved']}")

    return results

# test bullets
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewritten_bullets = rewrite_bullets(bullets)


# The original bullets are a bit weak, so the model suggested stronger action verbs and more specific language. However, some suggested outcomes should be verified to avoid adding achievements that were not explicitly stated in the original bullets.


# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés. 

    - Use only qualifications, experience, education, and skills explicitly provided by the user.
    - Never assume the user has a degree, completed coursework, built projects, or knows particular tools.
    - If information is missing, use a placeholder or ask a clarifying question instead of inventing details.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]
    # call get_completion() and return the result

    response = get_completion(messages)
    return response

# Test 
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Pandas."

cover_letter = generate_cover_letter(job_title, background)

print("\nGenerated Cover Letter Opening:")
print(cover_letter)

# I chose examples that demonstrate career changes into technical roles. They show how to connect previous industry experience with newly acquired technical skills. Few-shot prompting helps control the tone, structure, and specificity of the generated cover letter.


# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # return True if safe, False if flagged, and print a message if flagged

    if flagged:
        print("Your message was flagged. Please rephrase your request.")
        return False
    return True

# Test
safe_input = "Help me improve my resume for a data engineering position."
unsafe_input = "I want to kill someone."

print("Safe input:", is_safe(safe_input))
print("Unsafe input:", is_safe(unsafe_input))

# Borderline test

borderline_input = "I am extremely angry at my former employer."

result = client.moderations.create(
    model="omni-moderation-latest",
    input=borderline_input
)

print("Flagged:", result.results[0].flagged)
print("Categories:", result.results[0].categories)


# Task 5: The Chatbot Loop
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            if not raw_bullets:
                print("Job Application Helper: No bullet points were provided.")
                continue

            # Check the bullet points before sending them to the model
            if not all(is_safe(bullet) for bullet in raw_bullets):
                continue

            results = rewrite_bullets(raw_bullets)

            # Add the interaction to conversation history
            messages.append({
                "role": "user",
                "content": user_input + "\n" + "\n".join(raw_bullets)
            })

            messages.append({
                "role": "assistant",
                "content": json.dumps(results)
            })


        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # Check additional user inputs before sending them to the model
            if not is_safe(job_title) or not is_safe(background):
                continue

            cover_letter = generate_cover_letter(job_title, background)

            print("\nCover Letter Opening:")
            print(cover_letter)

            # Add the interaction to conversation history
            messages.append({
                "role": "user",
                "content": (
                    f"{user_input}\nJob title: {job_title}\n"
                    f"Background: {background}"
                )
            })

            messages.append({
                "role": "assistant",
                "content": cover_letter
            })

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({
                "role": "user",
                "content": user_input
            })

            response = get_completion(messages)

            print("\nJob Application Helper:", response)

            messages.append({
                "role": "assistant",
                "content": response
            })



if __name__ == "__main__":
    run_chatbot()



# Task 6: Ethics Reflection
# Option A — Comment block

# 1. The bot is trained on large amounts of text written by and about different people, so it can reflect biases present in that data. For example, it might favor certain communication styles, industries, or cultural backgrounds while treating others as less professional or suitable. This could lead to advice that does not represent every job-seeker fairly.

# 2. There are many things that can potentially be wrong about the bot's output, one of them is hallucination. It might contain invented information about your skills or experience and that would be your responsibility for providing a false information to a real employer - many further issues




