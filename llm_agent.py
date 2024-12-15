# llm_agent.py

from typing import Dict, List, Optional
from llm_providers import create_llm
import time  # Added this import
from prompts import (
    CODEMASTER_SYSTEM_PROMPT,
    GUESSER_SYSTEM_PROMPT,
    get_codemaster_prompt,
    get_guesser_prompt,
    GUESSER_FEEDBACK_PROMPT
)

class LLMAgent:
    def __init__(self, model_config: Dict):
        """Initialize an LLM agent with specific configuration"""
        self.llm = create_llm(model_config)
        self.role: Optional[str] = None

    def initialize_role(self, role: str):
        """Set the role for this LLM agent"""
        self.role = role
        self.system_prompt = (CODEMASTER_SYSTEM_PROMPT if role == 'codemaster' 
                            else GUESSER_SYSTEM_PROMPT)

    def _make_request(self, messages: List[Dict], max_tokens: int) -> str:
        """Make an API request with retries"""
        max_retries = 5
        base_delay = 2.0  # Increased base delay
        
        for attempt in range(max_retries):
            try:
                return self.llm.generate(messages, max_tokens)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with longer initial delay
                wait_time = base_delay * (2 ** attempt)
                print(f"API Error: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def give_clue(self, 
                team: str,
                team_words: List[str], 
                neutral_words: List[str],
                opponent_words: List[str],
                assassin: str,
                game_state: Dict) -> str:
        """Generate a clue as the Codemaster"""
        if self.role != 'codemaster':
            raise ValueError("This agent is not initialized as a Codemaster")

        prompt = get_codemaster_prompt(
            team=team,
            team_words=team_words,
            neutral_words=neutral_words,
            opponent_words=opponent_words,
            assassin=assassin,
            game_state=game_state
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return self._make_request(messages, max_tokens=20)

    def make_guess(self, 
                team: str,
                board: List[str], 
                clue: str, 
                number: int,
                game_state: Dict) -> str:
        """Make a guess as the Guesser"""
        if self.role != 'guesser':
            raise ValueError("This agent is not initialized as a Guesser")

        # Filter out already guessed words
        available_words = [word for word in board if word not in game_state["guessed_words"]]
        
        prompt = get_guesser_prompt(
            team=team,
            board=available_words,
            clue=clue,
            number=number,
            game_state=game_state
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        guess = self._make_request(messages, max_tokens=10)
        
        # Validate the guess is from available words
        if guess not in available_words:
            guess = available_words[0] if available_words else board[0]
        
        return guess

    def receive_guess_feedback(self,
                             guess: str,
                             result: str,
                             clue: str,
                             number: int,
                             game_state: Dict) -> None:
        """Process feedback about a guess"""
        if self.role != 'guesser':
            raise ValueError("This agent is not initialized as a Guesser")

        prompt = GUESSER_FEEDBACK_PROMPT.format(
            guess=guess,
            result=result,
            successful_guesses=game_state.get("current_turn_successes", []),
            unsuccessful_guesses=game_state.get("current_turn_failures", []),
            remaining_guesses=game_state.get("guesses_remaining", 0),
            clue=clue,
            number=number
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Don't need to wait for or process the response
        self._make_request(messages, max_tokens=10)