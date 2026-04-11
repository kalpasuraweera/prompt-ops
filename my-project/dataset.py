import json
import requests

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

PROMPT_FILE = "prompts/prompt.txt"
ATTACK_DATASET_FILE = "data/dataset_d.json"
OUTPUT_FILE = "data/dataset.json"

NUM_NORMAL_QUERIES = 6


# =========================
# OLLAMA CALL
# =========================
def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"].strip()


# =========================
# LOAD FILES
# =========================
def load_system_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_attack_dataset():
    with open(ATTACK_DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# GENERATE QUESTIONS
# =========================
def generate_questions(system_prompt, n):
    prompt = f"""
You are generating realistic user queries.

System prompt:
{system_prompt}

Generate {n} different normal user questions that match the assistant's role.

Rules:
- One question per line
- No numbering
- No explanations
- No JSON
- No extra text
"""

    output = call_ollama(prompt)

    # Parse line-by-line safely
    lines = []
    for line in output.split("\n"):
        line = line.strip()
        if len(line) > 5:
            lines.append(line)

    return lines[:n]


# =========================
# GENERATE ANSWER
# =========================
def generate_answer(system_prompt, question):
    prompt = f"""
System:
{system_prompt}

User:
{question}

Answer the user helpfully according to the system prompt.
"""

    return call_ollama(prompt)


# =========================
# BUILD DATASET
# =========================
def build_dataset():
    print("Building Dataset");
    system_prompt = load_system_prompt()
    attack_data = load_attack_dataset()

    final_data = attack_data.copy()

    # Generate normal queries
    questions = generate_questions(system_prompt, NUM_NORMAL_QUERIES)
    print("Qusestions Generated")

    for q in questions:
        answer = generate_answer(system_prompt, q)
        print("Answers Generated")

        final_data.append({
            "question": q,
            "answer": answer
        })

    return final_data


# =========================
# SAVE
# =========================
def save_dataset(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    dataset = build_dataset()
    save_dataset(dataset)
    print(f"✅ Dataset saved to {OUTPUT_FILE}")
