"""
Run multiple experiments and compute average scores and defection rates.
"""

import argparse
import subprocess
import sys
import pandas as pd


def run_experiments(num_runs: int, model: str, temperature: float) -> pd.DataFrame:
    """Run the experiment multiple times and collect results."""
    all_dfs = []
    
    for run in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"Run {run}/{num_runs}")
        print('='*60)
        
        cmd = [
            sys.executable,
            "pd/run_qwen_vs_fixed_strategies.py",
            "--model", model,
            "--temperature", str(temperature),
        ]
        
        subprocess.run(cmd, check=True)
        
        df = pd.read_csv("experiment_qwen_vs_fixed.csv")
        df["run"] = run
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)


def compute_statistics(df: pd.DataFrame) -> None:
    """Compute and print average scores and defection rates."""
    
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print('='*60)
    
    opponents = df["player2"].unique()
    
    results = []
    for opponent in opponents:
        opponent_df = df[df["player2"] == opponent]
        
        # Get final scores for each game (last round of each run)
        final_rounds = opponent_df[opponent_df["round"] == opponent_df["round"].max()]
        avg_agent_score = final_rounds["total1"].mean()
        avg_opponent_score = final_rounds["total2"].mean()
        
        # Calculate defection rate (F = defect)
        total_moves = len(opponent_df)
        agent_defections = (opponent_df["answer1"] == "F").sum()
        opponent_defections = (opponent_df["answer2"] == "F").sum()
        
        agent_defection_rate = agent_defections / total_moves
        opponent_defection_rate = opponent_defections / total_moves
        
        results.append({
            "opponent": opponent,
            "avg_agent_score": avg_agent_score,
            "avg_opponent_score": avg_opponent_score,
            "agent_defection_rate": agent_defection_rate,
            "opponent_defection_rate": opponent_defection_rate,
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nAverage Scores (per game):")
    print("-" * 40)
    for _, row in results_df.iterrows():
        print(f"  vs {row['opponent']:15s}: Agent={row['avg_agent_score']:.1f}, Opponent={row['avg_opponent_score']:.1f}")
    
    print("\nDefection Rates:")
    print("-" * 40)
    for _, row in results_df.iterrows():
        print(f"  vs {row['opponent']:15s}: Agent={row['agent_defection_rate']:.1%}, Opponent={row['opponent_defection_rate']:.1%}")
    
    # Save aggregate results
    output_path = "experiment_aggregate_stats.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nAggregate statistics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multiple PD experiments and compute statistics")
    parser.add_argument("--runs", type=int, default=5, help="Number of experiment runs (default: 5)")
    parser.add_argument("--model", type=str, default="qwen/qwen2.5-3b-instruct", help="Model to use")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for LLM sampling (default: 1)")
    args = parser.parse_args()
    
    df = run_experiments(args.runs, args.model, args.temperature)
    
    # Save all runs
    all_runs_path = "experiment_all_runs.csv"
    df.to_csv(all_runs_path, index=False)
    print(f"\nAll runs saved to {all_runs_path}")
    
    compute_statistics(df)


if __name__ == "__main__":
    main()
