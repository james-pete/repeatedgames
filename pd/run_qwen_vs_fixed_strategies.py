"""
Reproduce Prisoner's Dilemma experiments from "Playing repeated games with Large Language Models"
Agent: Qwen 2.5-3B-Instruct (via RunPod API)
Opponents: defect, cooperate, defect-once, tit-for-tat
"""

import os
import time
import pandas as pd
from openai import OpenAI

# --- Configuration ---
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY")
RUNPOD_ENDPOINT_URL = os.environ.get("RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen/qwen2.5-3b-instruct")

NUM_ROUNDS = 10
NUM_REPETITIONS = 1  # Number of times to repeat each matchup

# --- RunPod Client Setup ---
client = OpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=RUNPOD_ENDPOINT_URL,
)

# --- Fixed Strategies ---

def act_defect(text: str, round_num: int, opponent_last_move: str) -> str:
    """Always defect (Option F)."""
    return "F"

def act_cooperate(text: str, round_num: int, opponent_last_move: str) -> str:
    """Always cooperate (Option J)."""
    return "J"

def act_defect_once(text: str, round_num: int, opponent_last_move: str) -> str:
    """Defect on round 1, then cooperate."""
    return "F" if round_num == 1 else "J"

def act_tit_for_tat(text: str, round_num: int, opponent_last_move: str) -> str:
    """Cooperate on round 1, then mirror opponent's last move."""
    if round_num == 1:
        return "J"
    return opponent_last_move


# --- LLM Agent ---

def act_qwen(text: str, round_num: int, opponent_last_move: str, max_retries: int = 5) -> str:
    """Query Qwen 2.5-3B-Instruct via RunPod API."""
    messages = [{"role": "user", "content": text}]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=1,
                temperature=1.0,
                messages=messages,
            )
            answer = response.choices[0].message.content.strip()
            if answer in ["J", "F"]:
                return answer
            # If response isn't J or F, try to extract it
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


# --- Game Logic ---

def calculate_points(move1: str, move2: str) -> tuple[int, int]:
    """Calculate payoffs for a round of Prisoner's Dilemma."""
    payoff_matrix = {
        ("J", "J"): (8, 8),   # Both cooperate
        ("J", "F"): (0, 10),  # Player 1 cooperates, Player 2 defects
        ("F", "J"): (10, 0),  # Player 1 defects, Player 2 cooperates
        ("F", "F"): (5, 5),   # Both defect
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


def play_game(agent_fn, opponent_fn, opponent_name: str) -> list[dict]:
    """Play a full game between agent and opponent."""
    data = []
    agent_history = ""
    opponent_history = ""
    agent_last_move = ""
    opponent_last_move = ""
    total_agent = 0
    total_opponent = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        agent_prompt = build_prompt(agent_history, round_num)
        opponent_prompt = build_prompt(opponent_history, round_num)

        agent_move = agent_fn(agent_prompt, round_num, opponent_last_move)
        opponent_move = opponent_fn(opponent_prompt, round_num, agent_last_move)

        agent_points, opponent_points = calculate_points(agent_move, opponent_move)
        total_agent += agent_points
        total_opponent += opponent_points

        agent_history += (
            f"In round {round_num}, you chose Option {agent_move} and the other player "
            f"chose Option {opponent_move}. Thus, you won {agent_points} points and "
            f"the other player won {opponent_points} points.\n"
        )
        opponent_history += (
            f"In round {round_num}, you chose Option {opponent_move} and the other player "
            f"chose Option {agent_move}. Thus, you won {opponent_points} points and "
            f"the other player won {agent_points} points.\n"
        )

        agent_last_move = agent_move
        opponent_last_move = opponent_move

        data.append({
            "round": round_num,
            "player1": MODEL_NAME,
            "player2": opponent_name,
            "answer1": agent_move,
            "answer2": opponent_move,
            "points1": agent_points,
            "points2": opponent_points,
            "total1": total_agent,
            "total2": total_opponent,
        })

        print(f"  Round {round_num}: Agent={agent_move}, {opponent_name}={opponent_move} "
              f"| Points: {agent_points}-{opponent_points} | Total: {total_agent}-{total_opponent}")

    return data


def main():
    opponents = [
        (act_defect, "act_defect"),
        (act_cooperate, "act_cooperate"),
        (act_defect_once, "act_defect_once"),
        (act_tit_for_tat, "act_tit_for_tat"),
    ]

    all_data = []

    for rep in range(1, NUM_REPETITIONS + 1):
        print(f"\n{'='*60}")
        print(f"Repetition {rep}/{NUM_REPETITIONS}")
        print('='*60)

        for opponent_fn, opponent_name in opponents:
            print(f"\n--- {MODEL_NAME} vs {opponent_name} ---")
            game_data = play_game(act_qwen, opponent_fn, opponent_name)
            for row in game_data:
                row["repetition"] = rep
            all_data.extend(game_data)

    df = pd.DataFrame(all_data)
    output_path = "experiment_qwen_vs_fixed.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print('='*60)

    # Print summary
    print("\nSummary (average total points per game):")
    summary = df.groupby("player2").agg({
        "total1": lambda x: x.iloc[-1] if len(x) > 0 else 0,
        "total2": lambda x: x.iloc[-1] if len(x) > 0 else 0,
    }).reset_index()
    summary.columns = ["Opponent", "Agent Total", "Opponent Total"]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
