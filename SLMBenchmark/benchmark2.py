from pathlib import Path
from datetime import datetime
import re
import time

import ollama
import pandas as pd

# ============================================================
# 1. EXPERIMENT CONFIGURATION & ANSWER KEY
# ============================================================

MODELS = [
    "qwen2.5:3b",
    "phi4-mini",
    "llama3.2:3b",
]

TEMPERATURE = 0
TOP_P = 1.0
CONTEXT_LENGTH = 2048  # Reduced since we evaluate 1 problem at a time
MAX_OUTPUT_TOKENS = 512  # Focused output space per problem
SEED = 42

NUMBER_OF_RUNS = 1

# --- PATH SETTINGS ---
BASE_DIR = Path(__file__).parent
PROMPT_FILE = BASE_DIR / "prompt2.txt"
RESPONSES_DIR = BASE_DIR / "responses"
RESULTS_FILE = BASE_DIR / "benchmark_results2.csv"

# Updated Answer Key for Questions 1 through 25
ANSWER_KEY = {
    1: "A", 2: "C", 3: "D", 4: "E", 5: "B",
    6: "E", 7: "C", 8: "D", 9: "B", 10: "A",
    11: "B", 12: "D", 13: "A", 14: "B", 15: "A",
    16: "D", 17: "A", 18: "B", 19: "D", 20: "D",
    21: "B", 22: "A", 23: "C", 24: "E", 25: "E"
}

# System Instructions prepended to each individual problem prompt
SYSTEM_INSTRUCTIONS = """You are a mathematical reasoning assistant. 
Solve the given problem step-by-step. Show a concise 1-3 sentence explanation of your working, then end your response with the final answer strictly in this format:

Final Answer: (LETTER)

Do not include any extra text after the final answer choice.
"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_problems_from_prompt(full_prompt_text: str) -> dict:
    """
    Extracts individual problem blocks (Problem 1 through Problem 25)
    from the raw exam prompt text.
    """
    problems = {}
    # Matches "Problem 1", "Problem 2", up to next problem or end of string
    pattern = r"(Problem\s+(\d+)[\s\S]*?)(?=(?:Problem\s+\d+|$))"
    matches = re.findall(pattern, full_prompt_text, re.IGNORECASE)

    for block, q_num in matches:
        problems[int(q_num)] = block.strip()

    return problems


def extract_letter_answer(text: str) -> str:
    """
    Extracts the final letter choice (A, B, C, D, or E) from model response.
    """
    matches = re.findall(r"Final Answer:\s*[\(]?([A-E])[\)]?", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Fallback search if strict format was missed
    fallback = re.findall(r"\b([A-E])\b", text)
    return fallback[-1].upper() if fallback else "N/A"


# ============================================================
# 2. PREPARE INPUTS AND OUTPUT FOLDER
# ============================================================

RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

if not PROMPT_FILE.exists():
    raise FileNotFoundError(f"prompt2.txt was not found at {PROMPT_FILE}")

full_prompt_text = PROMPT_FILE.read_text(encoding="utf-8").strip()
problem_dict = parse_problems_from_prompt(full_prompt_text)

if len(problem_dict) == 0:
    print("Warning: Regex couldn't auto-split problems. Falling back to all 25 questions in key.")
    # Fallback if text format differs
    problem_dict = {q: f"Problem {q}\n" + full_prompt_text for q in ANSWER_KEY.keys()}

print(f"Loaded {len(problem_dict)} individual problems from {PROMPT_FILE.name}")

# ============================================================
# 3. RUN THE BENCHMARK (PROBLEM-BY-PROBLEM LOOP)
# ============================================================

results = []

for model in MODELS:
    for run_number in range(1, NUMBER_OF_RUNS + 1):

        print("\n" + "=" * 70)
        print(f"Running model: {model} | Run {run_number}/{NUMBER_OF_RUNS}")
        print("=" * 70)

        correct_count = 0
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_generation_time_s = 0.0

        full_run_transcript = []

        start_run_time = time.perf_counter()

        for q_num in sorted(ANSWER_KEY.keys()):
            problem_text = problem_dict.get(q_num, f"Problem {q_num} text not found.")
            expected_answer = ANSWER_KEY[q_num]

            single_problem_prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{problem_text}"

            try:
                t0 = time.perf_counter()
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": single_problem_prompt}],
                    stream=False,
                    options={
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "num_ctx": CONTEXT_LENGTH,
                        "num_predict": MAX_OUTPUT_TOKENS,
                        "seed": SEED,
                    },
                )
                dt = time.perf_counter() - t0

                output_text = response["message"]["content"].strip()
                extracted_choice = extract_letter_answer(output_text)
                is_correct = (extracted_choice == expected_answer.upper())

                if is_correct:
                    correct_count += 1

                p_tokens = response.get("prompt_eval_count", 0) or 0
                o_tokens = response.get("eval_count", 0) or 0
                gen_duration_ns = response.get("eval_duration", 0) or 0

                total_prompt_tokens += p_tokens
                total_output_tokens += o_tokens
                total_generation_time_s += (gen_duration_ns / 1_000_000_000)

                status_symbol = "✓" if is_correct else "✗"
                print(
                    f"Q{q_num:02d}: Model Choice = [{extracted_choice}] | Key = [{expected_answer}] {status_symbol} ({dt:.2f}s)")

                full_run_transcript.append(
                    f"--- QUESTION {q_num} ---\n{problem_text}\n\n"
                    f"MODEL RESPONSE:\n{output_text}\n\n"
                    f"EXTRACTED: {extracted_choice} | EXPECTED: {expected_answer} | RESULT: {status_symbol}\n\n"
                )

            except Exception as e:
                print(f"Q{q_num:02d}: Failed - Error: {e}")
                full_run_transcript.append(f"--- QUESTION {q_num} ---\nERROR: {e}\n\n")

        total_wall_clock = time.perf_counter() - start_run_time
        accuracy_pct = round((correct_count / len(ANSWER_KEY)) * 100, 2)
        total_tokens = total_prompt_tokens + total_output_tokens

        tok_per_sec = (
            round(total_output_tokens / total_generation_time_s, 2)
            if total_generation_time_s > 0
            else 0.0
        )

        # Save individual response log
        safe_model = model.replace(":", "_").replace("/", "_")
        log_file = RESPONSES_DIR / f"{safe_model}_loop_run_{run_number}.txt"
        log_file.write_text("\n".join(full_run_transcript), encoding="utf-8")

        results.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "run": run_number,
            "status": "success",
            "correct_answers": correct_count,
            "total_questions": len(ANSWER_KEY),
            "accuracy_pct": accuracy_pct,
            "wall_clock_latency_s": round(total_wall_clock, 2),
            "generation_duration_s": round(total_generation_time_s, 2),
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tok_per_sec,
            "response_file": str(log_file),
        })

        print("-" * 70)
        print(f"Completed {model}: Score = {correct_count}/{len(ANSWER_KEY)} ({accuracy_pct}%)")
        print(f"Total Time: {total_wall_clock:.2f}s | Output Speed: {tok_per_sec} tok/s")
        print(f"Detailed log saved to: {log_file}")

# ============================================================
# 4. SAVE SUMMARY RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8")

print("\n" + "=" * 70)
print(f"All runs completed! Results exported to {RESULTS_FILE}")
print("=" * 70)