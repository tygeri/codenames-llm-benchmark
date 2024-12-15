from typing import Dict, Tuple, List
from game_logger import GameLogger
from llm_agent import LLMAgent
import random

class CodeNamesBenchmark:
    def __init__(self, log_dir: str = "game_logs"):
        self.metrics = {}
        self.logger = GameLogger(log_dir)

    def simulate_game(self, game_id: int, team_a_config: Dict, team_b_config: Dict) -> Dict:
        """Simulate a game with 4 LLM instances (2v2)."""
        # Initialize 4 separate LLM agents
        team_a_codemaster = LLMAgent(team_a_config)
        team_a_guesser = LLMAgent(team_a_config)
        team_b_codemaster = LLMAgent(team_b_config)
        team_b_guesser = LLMAgent(team_b_config)

        # Set their roles
        team_a_codemaster.initialize_role("codemaster")
        team_a_guesser.initialize_role("guesser")
        team_b_codemaster.initialize_role("codemaster")
        team_b_guesser.initialize_role("guesser")

        # Generate board and assign words
        board = self.generate_board()
        team_a_words, team_b_words, neutral_words, assassin = self.split_words(board)

        # Initialize game state with proper tracking of past turns
        game_state = {
            "guessed_words": set(),  # All words that have been guessed
            "current_turn_guesses": [],  # Guesses made in current turn
            "guesses_remaining": 0,
            "past_turns": [],  # Detailed history of all turns
        }

        current_team = "A"  # Team A starts
        turn_count = 0
        game_over = False
        winner = None

        # Track metrics
        team_metrics = {
            "A": {"correct_guesses": 0, "incorrect_guesses": 0, "total_clues": 0},
            "B": {"correct_guesses": 0, "incorrect_guesses": 0, "total_clues": 0}
        }

        print(f"\nStarting game {game_id}")
        self.display_board(board, game_state["guessed_words"])

        while not game_over and turn_count < 20:  # Max 20 turns for safety
            turn_count += 1
            current_words = team_a_words if current_team == "A" else team_b_words
            opposing_words = team_b_words if current_team == "A" else team_a_words
            
            # Check if current team has any words left
            if not current_words:
                game_over = True
                winner = current_team
                print(f"\nTeam {current_team} wins by finding all their words!")
                break

            # Reset turn state
            game_state["current_turn_guesses"] = []
            turn_guesses = []
            turn_results = []

            # Get current team's agents
            current_codemaster = team_a_codemaster if current_team == "A" else team_b_codemaster
            current_guesser = team_a_guesser if current_team == "A" else team_b_guesser

            print(f"\n=== Team {current_team}'s Turn (Turn {turn_count}) ===")
            print(f"Remaining words to guess: {', '.join(current_words)}")

            # Codemaster gives clue
            try:
                clue = current_codemaster.give_clue(
                    f"Team {current_team}",
                    current_words, 
                    neutral_words, 
                    opposing_words, 
                    assassin, 
                    game_state
                )
                clue_word, clue_number = clue.split('\n')
                clue_number = int(clue_number)
                print(f"Codemaster's clue: {clue_word} {clue_number}")
            except (ValueError, TypeError) as e:
                print(f"Invalid clue format: {e}")
                current_team = "B" if current_team == "A" else "A"
                continue

            team_metrics[current_team]["total_clues"] += 1
            remaining_guesses = clue_number + 1
            game_state["guesses_remaining"] = remaining_guesses

            # Guesser makes guesses
            while remaining_guesses > 0 and not game_over:
                guess = current_guesser.make_guess(
                    f"Team {current_team}",
                    board, 
                    clue_word, 
                    clue_number, 
                    game_state
                )
                
                game_state["current_turn_guesses"].append(guess)
                turn_guesses.append(guess)
                print(f"Guesser's guess: {guess}")

                # Process guess and record result
                if guess in current_words:
                    print(f"Correct guess! Found a team word.")
                    team_metrics[current_team]["correct_guesses"] += 1
                    current_words.remove(guess)
                    game_state["guessed_words"].add(guess)
                    turn_results.append("team word")

                    if not current_words:  # Win condition
                        game_over = True
                        winner = current_team
                        print(f"\nTeam {current_team} wins by finding all their words!")
                        break

                elif guess == assassin:
                    print(f"Oh no! Hit the assassin word!")
                    team_metrics[current_team]["incorrect_guesses"] += 1
                    turn_results.append("assassin")
                    game_over = True
                    winner = "B" if current_team == "A" else "A"
                    break

                elif guess in opposing_words:
                    print(f"Oops! Found opponent's word.")
                    team_metrics[current_team]["incorrect_guesses"] += 1
                    opposing_words.remove(guess)
                    game_state["guessed_words"].add(guess)
                    turn_results.append("opponent word")
                    break

                elif guess in neutral_words:
                    print(f"Hit a neutral word.")
                    team_metrics[current_team]["incorrect_guesses"] += 1
                    neutral_words.remove(guess)
                    game_state["guessed_words"].add(guess)
                    turn_results.append("neutral")
                    break

                remaining_guesses -= 1
                game_state["guesses_remaining"] = remaining_guesses
                
                if remaining_guesses > 0:
                    print(f"Remaining guesses this turn: {remaining_guesses}")

            # Record turn in game history
            game_state["past_turns"].append({
                "turn_number": turn_count,
                "team": f"Team {current_team}",
                "clue_word": clue_word,
                "clue_number": clue_number,
                "guesses": turn_guesses,
                "results": turn_results
            })

            # Display board state after the turn
            self.display_board(board, game_state["guessed_words"])

            # Switch teams if game isn't over
            if not game_over:
                current_team = "B" if current_team == "A" else "A"

        # Return game results...
        return {
            "team_a": {
                "correct_guesses": team_metrics["A"]["correct_guesses"],
                "incorrect_guesses": team_metrics["A"]["incorrect_guesses"],
                "words_per_clue": (team_metrics["A"]["correct_guesses"] / 
                                team_metrics["A"]["total_clues"] if team_metrics["A"]["total_clues"] else 0),
                "won": winner == "A"
            },
            "team_b": {
                "correct_guesses": team_metrics["B"]["correct_guesses"],
                "incorrect_guesses": team_metrics["B"]["incorrect_guesses"],
                "words_per_clue": (team_metrics["B"]["correct_guesses"] / 
                                team_metrics["B"]["total_clues"] if team_metrics["B"]["total_clues"] else 0),
                "won": winner == "B"
            }
        }

    def _get_game_results(self, correct_guesses: int, incorrect_guesses: int, 
                            total_clues: int, won: bool) -> Dict:
        return {
            "correct_guesses": correct_guesses,
            "incorrect_guesses": incorrect_guesses,
            "words_per_clue": correct_guesses / total_clues if total_clues else 0,
            "won": won,
        }

    def generate_board(self) -> List[str]:
        """
        Generate a random board with words.
        """
        with open("words/default.txt", "r") as file:  # Changed path
            words = file.read().splitlines()
        return random.sample(words, 25)

    def split_words(self, board: List[str]) -> Tuple[List[str], List[str], List[str], str]:
        """
        Split the board into team, neutral, opponent, and assassin words.
        """
        random.shuffle(board)
        return board[:9], board[9:17], board[17:24], board[24]

    def display_board(self, board, guessed_words):
        print("\nBoard State:")
        for i in range(5):
            row = board[i * 5:(i + 1) * 5]
            display = [f"[{word}]" if word in guessed_words else word for word in row]
            print(" | ".join(display))
        print()

    def run_matchup(self,
                    team_a_config: Dict,
                    team_b_config: Dict,
                    num_games: int) -> Dict:
        """Run a series of games between two teams"""
        results = {
            "team_a": {
                "model": team_a_config["model_name"],  # Changed from "name" to "model_name"
                "games_played": 0,
                "wins": 0,
                "total_correct_guesses": 0,
                "total_incorrect_guesses": 0,
                "average_words_per_clue": []
            },
            "team_b": {
                "model": team_b_config["model_name"],  # Changed from "name" to "model_name"
                "games_played": 0,
                "wins": 0,
                "total_correct_guesses": 0,
                "total_incorrect_guesses": 0,
                "average_words_per_clue": []
            }
        }

        for i in range(num_games):
            game_results = self.simulate_game(i, team_a_config, team_b_config)
            
            # Update team A stats
            results["team_a"]["games_played"] += 1
            results["team_a"]["wins"] += 1 if game_results["team_a"]["won"] else 0
            results["team_a"]["total_correct_guesses"] += game_results["team_a"]["correct_guesses"]
            results["team_a"]["total_incorrect_guesses"] += game_results["team_a"]["incorrect_guesses"]
            results["team_a"]["average_words_per_clue"].append(game_results["team_a"]["words_per_clue"])

            # Update team B stats
            results["team_b"]["games_played"] += 1
            results["team_b"]["wins"] += 1 if game_results["team_b"]["won"] else 0
            results["team_b"]["total_correct_guesses"] += game_results["team_b"]["correct_guesses"]
            results["team_b"]["total_incorrect_guesses"] += game_results["team_b"]["incorrect_guesses"]
            results["team_b"]["average_words_per_clue"].append(game_results["team_b"]["words_per_clue"])

        # Calculate final averages
        for team in ["team_a", "team_b"]:
            results[team]["win_rate"] = results[team]["wins"] / results[team]["games_played"]
            results[team]["average_words_per_clue"] = (
                sum(results[team]["average_words_per_clue"]) / len(results[team]["average_words_per_clue"])
            )

        self.metrics = results
        return results
    
# adding a change