"""
Conversation Management for Fleming-AI Review System
Manages multi-turn dialogue between Fleming and Alex reviewers
"""

from datetime import datetime
from typing import Optional
from src.reviewers.schemas import ReviewTurn, ConversationState


class ConversationManager:
    """
    Manages multi-turn dialogue between Fleming and Alex.

    Tracks conversation state, detects convergence, and identifies escalation conditions.
    """

    def __init__(self, max_turns: int = 6, max_tokens_budget: int = 50000):
        """Initialize conversation manager.

        Args:
            max_turns: Maximum dialogue turns before forcing convergence (default: 6)
            max_tokens_budget: Max tokens in history before truncation (default: 50K)
        """
        self.max_turns = max_turns
        self.max_tokens_budget = max_tokens_budget
        self.state = ConversationState(
            session_id=f"conv_{datetime.now().timestamp()}",
            stage="unknown",
            turns=[],
            current_score=None,
            is_converged=False,
        )
        self._score_history = []  # Track score progression for escalation detection

    def add_turn(self, speaker: str, content: str, structured: dict):
        """Add a turn to the conversation.

        Args:
            speaker: 'fleming' or 'alex'
            content: Text content of the turn
            structured: Structured data (e.g., ReviewResult dict for Alex, revised artifact for Fleming)
        """
        turn = ReviewTurn(
            turn_id=f"turn_{len(self.state.turns) + 1}",
            speaker=speaker,
            content=content,
            structured_data=structured,
            timestamp=datetime.now().isoformat(),
        )
        self.state.turns.append(turn)

        # Update score if this is Alex's review
        if speaker == "alex" and "scores" in structured:
            scores = structured["scores"]
            # Calculate average score across all dimensions
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                self.state.current_score = avg_score
                self._score_history.append(avg_score)

        # Check convergence after adding turn
        if speaker == "alex" and structured.get("verdict") == "PASS":
            self.state.is_converged = True

        if len(self.state.turns) >= self.max_turns:
            self.state.is_converged = True

    def get_context_for_prompt(self) -> str:
        """Format conversation history for inclusion in LLM prompt.

        Returns formatted string like:
        [Turn 1 - Fleming]: Hypothesis: "ViT is better..."
        [Turn 2 - Alex]: REVISE. Issues: no conditions. Suggest: add "when".
        [Turn 3 - Fleming]: Revised: "ViT is better when data < 25%..."
        [Turn 4 - Alex]: PASS. Improvements noted.
        """
        if not self.state.turns:
            return "No previous conversation."

        formatted = []
        for i, turn in enumerate(self.state.turns, 1):
            speaker_name = "Fleming" if turn.speaker == "fleming" else "Alex"

            # Truncate long content (keep first 500 chars)
            content_preview = turn.content[:500]
            if len(turn.content) > 500:
                content_preview += "..."

            # Add verdict for Alex's turns
            if turn.speaker == "alex" and "verdict" in turn.structured_data:
                verdict = turn.structured_data["verdict"]
                formatted.append(f"[Turn {i} - {speaker_name}]: {verdict}. {content_preview}")
            else:
                formatted.append(f"[Turn {i} - {speaker_name}]: {content_preview}")

        return "\n".join(formatted)

    def is_converged(self) -> bool:
        """Check if conversation has converged.

        Convergence conditions:
        1. Alex gave PASS verdict
        2. Max turns reached

        Returns:
            True if conversation has converged, False otherwise
        """
        return self.state.is_converged

    def should_escalate(self) -> Optional[str]:
        """Check if conversation should escalate (restart stage or pipeline).

        Escalation conditions:
        1. Last 2 Alex reviews show no score improvement → "No improvement in 2 turns"
        2. Alex returned RESTART_STAGE verdict → "Alex requested stage restart"
        3. Alex returned RESTART_PIPELINE verdict → "Alex requested pipeline restart"

        Returns:
            None if no escalation needed
            Reason string if escalation needed
        """
        if len(self.state.turns) < 2:
            return None

        # Check last Alex turn for explicit restart requests
        for turn in reversed(self.state.turns):
            if turn.speaker == "alex":
                verdict = turn.structured_data.get("verdict")
                if verdict == "RESTART_STAGE":
                    return "Alex requested stage restart"
                elif verdict == "RESTART_PIPELINE":
                    return "Alex requested pipeline restart"
                break

        # Check score progression (need at least 2 scores)
        if len(self._score_history) >= 2:
            last_two = self._score_history[-2:]
            if last_two[1] <= last_two[0]:
                # No improvement in last turn
                if len(self._score_history) >= 3:
                    last_three = self._score_history[-3:]
                    if last_three[2] <= last_three[1] <= last_three[0]:
                        # 2 consecutive turns without improvement
                        return "No score improvement in 2 consecutive turns"

        return None

    def truncate_if_needed(self):
        """Truncate old turns if token budget exceeded.

        Strategy: Keep most recent 2 turns, summarize older turns.
        Estimation: ~4 chars per token (rough approximation).
        """
        total_chars = sum(len(turn.content) for turn in self.state.turns)
        estimated_tokens = total_chars // 4

        if estimated_tokens > self.max_tokens_budget:
            # Keep last 2 turns
            recent_turns = self.state.turns[-2:]

            # Summarize older turns
            old_turn_count = len(self.state.turns) - 2
            summary_turn = ReviewTurn(
                turn_id="turn_summary",
                speaker="system",
                content=f"[{old_turn_count} earlier turns summarized]",
                structured_data={"type": "summary"},
                timestamp=datetime.now().isoformat(),
            )

            self.state.turns = [summary_turn] + recent_turns
