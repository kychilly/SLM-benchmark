from pathlib import Path
from datetime import datetime
import time

import ollama
import pandas as pd


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

MODELS = [
    "tinyllama:latest",
]

TEMPERATURE = 0
TOP_P = 1.0
CONTEXT_LENGTH = 2048  # Reduced to fit TinyLlama's maximum native context
MAX_OUTPUT_TOKENS = 1000
SEED = 42

NUMBER_OF_RUNS = 1

PROMPT_FILE = Path("prompt.txt")
RESPONSES_DIR = Path("responses")
RESULTS_FILE = Path("benchmark_results.csv")


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

            if generation_duration_s > 0:
                output_tokens_per_second = (
                    output_tokens / generation_duration_s
                )
            else:
                output_tokens_per_second = None

            safe_model_name = (
                model.replace(":", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )

            response_file = (
                RESPONSES_DIR
                / f"{safe_model_name}_run_{run_number}.txt"
            )

            response_file.write_text(
                output_text,
                encoding="utf-8",
            )

            results.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": model,
                    "run": run_number,
                    "status": "success",
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "context_length": CONTEXT_LENGTH,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "seed": SEED,
                    "wall_clock_latency_s": round(
                        wall_clock_latency, 4
                    ),
                    "ollama_total_duration_s": round(
                        total_duration_s, 4
                    ),
                    "model_load_duration_s": round(
                        load_duration_s, 4
                    ),
                    "prompt_processing_duration_s": round(
                        prompt_duration_s, 4
                    ),
                    "generation_duration_s": round(
                        generation_duration_s, 4
                    ),
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
            print(
                f"Wall-clock latency: "
                f"{wall_clock_latency:.2f} seconds"
            )
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Total tokens: {total_tokens}")
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
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "context_length": CONTEXT_LENGTH,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "seed": SEED,
                    "wall_clock_latency_s": round(
                        wall_clock_latency, 4
                    ),
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

results_df.to_csv(
    RESULTS_FILE,
    index=False,
    encoding="utf-8",
)

print("\n" + "=" * 70)
print(f"Detailed results saved to: {RESULTS_FILE}")
print("=" * 70)

print(results_df)


# ============================================================
# 6. CREATE SUMMARY TABLE
# ============================================================

successful_results = results_df[
    results_df["status"] == "success"
].copy()

if not successful_results.empty:
    summary_df = (
        successful_results
        .groupby("model", as_index=False)
        .agg(
            runs=("run", "count"),
            mean_latency_s=("wall_clock_latency_s", "mean"),
            std_latency_s=("wall_clock_latency_s", "std"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_total_tokens=("total_tokens", "mean"),
            mean_tokens_per_second=(
                "output_tokens_per_second",
                "mean",
            ),
        )
    )

    summary_df["mean_latency_s"] = (
        summary_df["mean_latency_s"].round(3)
    )
    summary_df["std_latency_s"] = (
        summary_df["std_latency_s"].round(3)
    )
    summary_df["mean_prompt_tokens"] = (
        summary_df["mean_prompt_tokens"].round(1)
    )
    summary_df["mean_output_tokens"] = (
        summary_df["mean_output_tokens"].round(1)
    )
    summary_df["mean_total_tokens"] = (
        summary_df["mean_total_tokens"].round(1)
    )
    summary_df["mean_tokens_per_second"] = (
        summary_df["mean_tokens_per_second"].round(3)
    )

    summary_df.to_csv(
        "benchmark_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nBenchmark summary:")
    print(summary_df)

else:
    print("No model completed successfully.")