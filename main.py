import os
from dotenv import load_dotenv
from benchmark import CodeNamesBenchmark
import json

def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("Missing API key. Please set OPENAI_API_KEY in .env file")

    # Configure the teams
    team_configs = {
        "gpt4_team": {
            "name": "gpt-4",
            "api_key": api_key,
            "temperature": 0.7
        },
        "gpt3_team": {
            "name": "gpt-3.5-turbo",
            "api_key": api_key,
            "temperature": 0.7
        }
    }

    # Initialize benchmark with logging
    benchmark = CodeNamesBenchmark(log_dir="game_logs")
    
    try:
        print("\n=== Starting 2v2 Match ===")
        print("Team A (GPT-4): Codemaster + Guesser")
        print("Team B (GPT-3.5): Codemaster + Guesser")
        
        metrics = benchmark.run_matchup(
            team_a_config=team_configs["gpt4_team"],
            team_b_config=team_configs["gpt3_team"],
            num_games=5
        )

        # Print results
        print("\n=== Match Results ===")
        for team, stats in metrics.items():
            print(f"\n{team.upper()} ({stats['model']}):")
            print(f"Wins: {stats['wins']}/{stats['games_played']}")
            print(f"Win Rate: {stats['win_rate']:.2%}")
            print(f"Correct guesses: {stats['total_correct_guesses']}")
            print(f"Incorrect guesses: {stats['total_incorrect_guesses']}")
            print(f"Average words per clue: {stats['average_words_per_clue']:.2f}")

    except Exception as e:
        print(f"Match failed: {e}")
        raise e  # Re-raise to see the full traceback

if __name__ == "__main__":
    main()