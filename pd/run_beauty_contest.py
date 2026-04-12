"""
Two-thirds average game.
Agent plays against 99 simulated players.
"""

import argparse
import os
import time
import random
import numpy as np
import pandas as pd
from openai import OpenAI

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY")
RUNPOD_ENDPOINT_URL = os.environ.get("RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1")

NUM_PLAYERS = 100
TARGET_FRACTION = 2/3

TEMPERATURE = 1
MODEL_NAME = "qwen/qwen2.5-3b-instruct"
client = None

SYSTEM_PROMPT = """
Respond in the following format, with only the numerical answer between the <answer> tags:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted response."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def build_prompt() -> str:
    """Build the game prompt for the LLM."""
    prompt = (
        f"You are playing a game with {NUM_PLAYERS - 1} other players "
        f"({NUM_PLAYERS} players total, including you).\n\n"
        "Rules:\n"
        "- Each player submits a number between 0 and 100 (inclusive).\n"
        f"- The winner is the player whose submission is closest to {TARGET_FRACTION:.4f} (two-thirds) "
        "of the average of all submissions.\n"
        "- Your goal is to be as close as possible to this target.\n"
        "- All other players are rational and have the same goal as you.\n\n"
        "Q: What number do you submit (between 0 and 100)?"
    )
    return prompt


def act_llm(text: str, max_retries: int = 5) -> float:
    """Query LLM via API and return a number."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]
    
    use_new_api = MODEL_NAME.startswith(("gpt-5", "o1", "o3"))
    
    for attempt in range(max_retries):
        try:
            if use_new_api:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    max_completion_tokens=4096,
                    messages=messages,
                )
            else:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=1024,
                    temperature=TEMPERATURE,
                    messages=messages,
                )
            raw_response = response.choices[0].message.content.strip()
            print(f"\n--- Model Response ---\n{raw_response}\n----------------------")
            
            answer_str = extract_xml_answer(raw_response)
            # Try to parse as a number
            answer = float(answer_str)
            # Clamp to valid range
            answer = max(0, min(100, answer))
            return answer
        except ValueError:
            print(f"Warning: Could not parse '{answer_str}' as number, retrying...")
        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    print("Warning: Failed to get valid response, defaulting to 50")
    return 50.0


def play_game() -> dict:
    """Play a single game."""
    prompt = build_prompt()
    print(f"\n--- Prompt ---\n{prompt}\n--------------")
    
    agent_submission = act_llm(prompt)
    print(f"\nAgent submitted: {agent_submission:.2f}")
    
    return {
        "agent_submission": agent_submission,
    }


def bootstrap_ci(data: list[float], n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean."""
    data = np.array(data)
    n = len(data)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(data)
    
    return mean, lower, upper


def main():
    global TEMPERATURE, MODEL_NAME, client
    
    parser = argparse.ArgumentParser(description="Run two-thirds average game")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for LLM sampling (default: 1)")
    parser.add_argument("--model", type=str, default="qwen/qwen2.5-3b-instruct", help="Model to use")
    parser.add_argument("--runs", type=int, default=1, help="Number of games to play (default: 1)")
    args = parser.parse_args()
    
    TEMPERATURE = args.temperature
    MODEL_NAME = args.model
    
    if MODEL_NAME.startswith("gpt"):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable must be set to use GPT models")
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = OpenAI(api_key=RUNPOD_API_KEY, base_url=RUNPOD_ENDPOINT_URL)
    
    all_results = []
    
    for run in range(1, args.runs + 1):
        print(f"\n{'='*60}")
        print(f"Game {run}/{args.runs}")
        print('='*60)
        
        result = play_game()
        result["run"] = run
        all_results.append(result)
    
    df = pd.DataFrame(all_results)
    
    # Summary statistics
    if args.runs > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        
        submissions = df['agent_submission'].tolist()
        mean, ci_lower, ci_upper = bootstrap_ci(submissions, n_bootstrap=2000, ci=0.95)
        std = np.std(submissions)
        
        print(f"\nAverage submission: {mean:.2f}")
        print(f"Std dev: {std:.2f}")
        print(f"95% CI (bootstrap): [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"\nSubmissions: {submissions}")
    
    output_path = "experiment_beauty_contest.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
