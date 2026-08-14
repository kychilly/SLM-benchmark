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
CONTEXT_LENGTH = 8192  # Expanded context to fit full 25-problem exam prompt
MAX_OUTPUT_TOKENS = 4096  # High token limit so the model can solve up to Problem 25
SEED = 42

NUMBER_OF_RUNS = 1

# --- PATH SETTINGS ---
BASE_DIR = Path(__file__).parent
PROMPT_FILE = BASE_DIR / "prompt2.txt"
RESPONSES_DIR = BASE_DIR / "responses"
RESULTS_FILE = BASE_DIR / "benchmark_results2.csv"

# Answer Key for Questions 1 through 25
ANSWER_KEY = {
    1: "B", 2: "D", 3: "D", 4: "A", 5: "E",
    6: "C", 7: "C", 8: "C", 9: "C", 10: "A",
    11: "C", 12: "C", 13: "D", 14: "B", 15: "D",
    16: "B", 17: "C", 18: "D", 19: "C", 20: "E",
    21: "C", 22: "C", 23: "E", 24: "D", 25: "E",
}


# ============================================================
# HELPER: EVALUATE MODEL OUTPUT AGAINST ANSWER KEY
# ============================================================

def evaluate_responses(text: str, answer_key: dict):
    """
    Parses output text for answer patterns matching formats like:
    - "The answer is (B):" or "The correct answer is B"
    - "Problem 1: (B)" or "Q1: B" or "1. (B)"
    Compares extracted choices against the provided answer key.
    """
    correct_count = 0
    total_questions = len(answer_key)

    for q_num, expected in answer_key.items():
        # Regex explanation:
        # 1. Finds question identifier: (Problem 1, Q1, 1.) OR flexible context around the question
        # 2. Captures explicit phrasing: "answer is (B)", "answer: B", etc.
        pattern = rf"(?:Problem|Q)?\s*{q_num}\b[\s\S]*?(?:answer\s+(?:is|:)?\s*[\(]?([A-E])[\)]?|[\.\:\)\s]+\(?([A-E])\)?\b)"

        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            # Flatten the tuple returned by re.findall groups and grab non-empty matches
            extracted_letters = [item for group in matches for item in group if item]
            if extracted_letters:
                predicted = extracted_letters[-1].upper()  # Takes the last explicitly stated answer for the question
                if predicted == expected.upper():
                    correct_count += 1

    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0.0
    return correct_count, round(accuracy, 2)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3. LOAD THE PROMPT
# ============================================================

if not PROMPT_FILE.exists():
    raise FileNotFoundError(
        "prompt.txt was not found. Create prompt.txt in the same folder "
        "as benchmark.py and paste the full exam prompt into it."
    )

prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()

if not prompt:
    raise ValueError("prompt.txt is empty.")

print(f"Prompt loaded successfully: {len(prompt):,} characters")

# ============================================================
# 4. RUN THE BENCHMARK
# ============================================================

results = []

for model in MODELS:
    for run_number in range(1, NUMBER_OF_RUNS + 1):

        print("\n" + "=" * 70)
        print(f"Running model: {model}")
        print(f"Run: {run_number}/{NUMBER_OF_RUNS}")
        print("=" * 70)

        start_time = time.perf_counter()

        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                stream=False,
                options={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "num_ctx": CONTEXT_LENGTH,
                    "num_predict": MAX_OUTPUT_TOKENS,
                    "seed": SEED,
                },
            )

            wall_clock_latency = time.perf_counter() - start_time
            output_text = response["message"]["content"]

            # Save the raw output response directly to a .txt file
            safe_model_name = (
                model.replace(":", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            response_file = RESPONSES_DIR / f"{safe_model_name}_run_{run_number}.txt"
            response_file.write_text(output_text, encoding="utf-8")

            # Score output against the embedded answer key
            correct_answers, accuracy_pct = evaluate_responses(output_text, ANSWER_KEY)

            prompt_tokens = response.get("prompt_eval_count", 0) or 0
            output_tokens = response.get("eval_count", 0) or 0

            total_duration_ns = response.get("total_duration", 0) or 0
            load_duration_ns = response.get("load_duration", 0) or 0
            prompt_duration_ns = response.get("prompt_eval_duration", 0) or 0
            generation_duration_ns = response.get("eval_duration", 0) or 0

            total_duration_s = total_duration_ns / 1_000_000_000
            load_duration_s = load_duration_ns / 1_000_000_000
            prompt_duration_s = prompt_duration_ns / 1_000_000_000
            generation_duration_s = generation_duration_ns / 1_000_000_000

            total_tokens = prompt_tokens + output_tokens

            output_tokens_per_second = (
                (output_tokens / generation_duration_s)
                if generation_duration_s > 0
                else None
            )

            results.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": model,
                    "run": run_number,
                    "status": "success",
                    "correct_answers": correct_answers,
                    "total_questions": len(ANSWER_KEY),
                    "accuracy_pct": accuracy_pct,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "context_length": CONTEXT_LENGTH,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "seed": SEED,
                    "wall_clock_latency_s": round(wall_clock_latency, 4),
                    "ollama_total_duration_s": round(total_duration_s, 4),
                    "model_load_duration_s": round(load_duration_s, 4),
                    "prompt_processing_duration_s": round(prompt_duration_s, 4),
                    "generation_duration_s": round(generation_duration_s, 4),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "output_tokens_per_second": (
                        round(output_tokens_per_second, 4)
                        if output_tokens_per_second is not None
                        else None
                    ),
                    "response_file": str(response_file),
                    "error": "",
                }
            )

            print(f"Completed: {model}")
            print(f"Accuracy: {correct_answers}/{len(ANSWER_KEY)} ({accuracy_pct}%)")
            print(f"Wall-clock latency: {wall_clock_latency:.2f} seconds")
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(
                f"Generation speed: "
                f"{output_tokens_per_second:.2f} tokens/second"
                if output_tokens_per_second is not None
                else "Generation speed unavailable"
            )
            print(f"Response saved to: {response_file}")

        except Exception as error:
            wall_clock_latency = time.perf_counter() - start_time
            print(f"Error while running {model}: {error}")

            results.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": model,
                    "run": run_number,
                    "status": "failed",
                    "correct_answers": 0,
                    "total_questions": len(ANSWER_KEY),
                    "accuracy_pct": 0.0,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "context_length": CONTEXT_LENGTH,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "seed": SEED,
                    "wall_clock_latency_s": round(wall_clock_latency, 4),
                    "ollama_total_duration_s": None,
                    "model_load_duration_s": None,
                    "prompt_processing_duration_s": None,
                    "generation_duration_s": None,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "output_tokens_per_second": None,
                    "response_file": "",
                    "error": str(error),
                }
            )

# ============================================================
# 5. SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8")

print("\n" + "=" * 70)
print(f"Detailed results saved to: {RESULTS_FILE}")
print("=" * 70)

# ============================================================
# 6. CREATE SUMMARY TABLE
# ============================================================

successful_results = results_df[results_df["status"] == "success"].copy()

if not successful_results.empty:
    summary_df = (
        successful_results
        .groupby("model", as_index=False)
        .agg(
            runs=("run", "count"),
            mean_accuracy_pct=("accuracy_pct", "mean"),
            mean_correct=("correct_answers", "mean"),
            mean_latency_s=("wall_clock_latency_s", "mean"),
            mean_tokens_per_second=("output_tokens_per_second", "mean"),
        )
    )

    summary_df["mean_accuracy_pct"] = summary_df["mean_accuracy_pct"].round(2)
    summary_df["mean_correct"] = summary_df["mean_correct"].round(1)
    summary_df["mean_latency_s"] = summary_df["mean_latency_s"].round(3)
    summary_df["mean_tokens_per_second"] = summary_df["mean_tokens_per_second"].round(3)

    summary_df.to_csv("benchmark_summary.csv", index=False, encoding="utf-8")

    print("\nBenchmark Summary:")
    print(summary_df)

else:
    print("No model completed successfully.")