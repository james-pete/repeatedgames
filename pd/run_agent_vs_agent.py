"""
Reproduce Prisoner's Dilemma experiments from "Playing repeated games with Large Language Models"
Two RunPod agents playing against each other.
"""

import os
import time
import pandas as pd
from openai import OpenAI

# --- Configuration ---
# Agent 1 (Player 1)
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY")
RUNPOD_ENDPOINT_URL = os.environ.get("RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1")
RUNPOD_MODEL_NAME = os.environ.get("RUNPOD_MODEL_NAME", "qwen/qwen2.5-3b-instruct")

# Agent 2 (Player 2 / Opponent)
RUNPOD_OPPONENT_API_KEY = os.environ.get("RUNPOD_OPPONENT_API_KEY", os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY"))
RUNPOD_OPPONENT_ENDPOINT_URL = os.environ.get("RUNPOD_OPPONENT_ENDPOINT_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1")
RUNPOD_OPPONENT_MODEL_NAME = os.environ.get("RUNPOD_OPPONENT_MODEL_NAME", "qwen/qwen2.5-3b-instruct")

NUM_ROUNDS = 10
NUM_REPETITIONS = 5

# --- RunPod Client Setup ---
client_agent1 = OpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=RUNPOD_ENDPOINT_URL,
)

client_agent2 = OpenAI(
    api_key=RUNPOD_OPPONENT_API_KEY,
    base_url=RUNPOD_OPPONENT_ENDPOINT_URL,
)


# --- LLM Agents ---

def query_llm(client: OpenAI, model_name: str, text: str, max_retries: int = 5) -> str:
    """Query an LLM via RunPod API."""
    messages = [{"role": "user", "content": text}]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=1,
                temperature=1.0,
                messages=messages,
            )
            answer = response.choices[0].message.content.strip()
            if answer in ["J", "F"]:
                return answer
            if "J" in answer:
                return "J"
            if "F" in answer:
                return "F"
            print(f"Warning: Unexpected response '{answer}', defaulting to J")
            return "J"
        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return "J"


def act_agent1(text: str, round_num: int, opponent_last_move: str) -> str:
    """Query Agent 1 via RunPod API."""
    return query_llm(client_agent1, RUNPOD_MODEL_NAME, text)


def act_agent2(text: str, round_num: int, opponent_last_move: str) -> str:
    """Query Agent 2 via RunPod API."""
    return query_llm(client_agent2, RUNPOD_OPPONENT_MODEL_NAME, text)


# --- Game Logic ---

def calculate_points(move1: str, move2: str) -> tuple[int, int]:
    """Calculate payoffs for a round of Prisoner's Dilemma."""
    payoff_matrix = {
        ("J", "J"): (8, 8),
        ("J", "F"): (0, 10),
        ("F", "J"): (10, 0),
        ("F", "F"): (5, 5),
    }
    return payoff_matrix.get((move1, move2), (-9999, -9999))


def build_prompt(conversation_history: str, round_num: int) -> str:
    """Build the game prompt for the LLM."""
    question = (
        "You are playing a game repeatedly with another player. "
        "In this game, you can choose between Option J and Option F. "
        f"You will play {NUM_ROUNDS} rounds in total with the same player. "
        "The rules of the game are as follows:\n"
        "If you choose Option J and the other player chooses Option J, then you win 8 points and the other player wins 8 points.\n"
        "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
        "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
        "If you choose Option F and the other player chooses Option F, then you win 5 points and the other player wins 5 points.\n\n"
    )
    prompt = (
        question + conversation_history +
        f"You are currently playing round {round_num}.\n"
        "Q: Which Option do you choose, Option J or Option F?\n"
        "Provide no reasoning and answer with a single character."
    )
    return prompt


def play_game(agent1_fn, agent2_fn, agent1_name: str, agent2_name: str) -> list[dict]:
    """Play a full game between two agents."""
    data = []
    agent1_history = ""
    agent2_history = ""
    total_agent1 = 0
    total_agent2 = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        agent1_prompt = build_prompt(agent1_history, round_num)
        agent2_prompt = build_prompt(agent2_history, round_num)

        agent1_move = agent1_fn(agent1_prompt, round_num, "")
        agent2_move = agent2_fn(agent2_prompt, round_num, "")

        agent1_points, agent2_points = calculate_points(agent1_move, agent2_move)
        total_agent1 += agent1_points
        total_agent2 += agent2_points

        agent1_history += (
            f"In round {round_num}, you chose Option {agent1_move} and the other player "
            f"chose Option {agent2_move}. Thus, you won {agent1_points} points and "
            f"the other player won {agent2_points} points.\n"
        )
        agent2_history += (
            f"In round {round_num}, you chose Option {agent2_move} and the other player "
            f"chose Option {agent1_move}. Thus, you won {agent2_points} points and "
            f"the other player won {agent1_points} points.\n"
        )

        data.append({
            "round": round_num,
            "player1": agent1_name,
            "player2": agent2_name,
            "answer1": agent1_move,
            "answer2": agent2_move,
            "points1": agent1_points,
            "points2": agent2_points,
            "total1": total_agent1,
            "total2": total_agent2,
        })

        print(f"  Round {round_num}: Agent1={agent1_move}, Agent2={agent2_move} "
              f"| Points: {agent1_points}-{agent2_points} | Total: {total_agent1}-{total_agent2}")

    return data


def main():
    print(f"Agent 1: {RUNPOD_MODEL_NAME}")
    print(f"Agent 2: {RUNPOD_OPPONENT_MODEL_NAME}")
    print(f"Rounds per game: {NUM_ROUNDS}")
    print(f"Repetitions: {NUM_REPETITIONS}")

    all_data = []

    for rep in range(1, NUM_REPETITIONS + 1):
        print(f"\n{'='*60}")
        print(f"Repetition {rep}/{NUM_REPETITIONS}")
        print('='*60)

        print(f"\n--- {RUNPOD_MODEL_NAME} vs {RUNPOD_OPPONENT_MODEL_NAME} ---")
        game_data = play_game(act_agent1, act_agent2, RUNPOD_MODEL_NAME, RUNPOD_OPPONENT_MODEL_NAME)
        for row in game_data:
            row["repetition"] = rep
        all_data.extend(game_data)

    df = pd.DataFrame(all_data)
    output_path = "experiment_agent_vs_agent.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print('='*60)

    # Print summary
    print("\nSummary (total points per repetition):")
    summary = df.groupby("repetition").agg({
        "total1": "last",
        "total2": "last",
    }).reset_index()
    summary.columns = ["Repetition", "Agent 1 Total", "Agent 2 Total"]
    print(summary.to_string(index=False))

    print(f"\nAverage scores across {NUM_REPETITIONS} games:")
    print(f"  Agent 1 ({RUNPOD_MODEL_NAME}): {summary['Agent 1 Total'].mean():.1f}")
    print(f"  Agent 2 ({RUNPOD_OPPONENT_MODEL_NAME}): {summary['Agent 2 Total'].mean():.1f}")


if __name__ == "__main__":
    main()
