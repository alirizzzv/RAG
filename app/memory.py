"""Per-session conversation memory.

Stores Q&A turns in a plain list. The Chainlit app owns the session dict;
this module just formats history into a prompt-friendly string so the answer
and code nodes can give context-aware follow-up responses.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    question: str
    answer: str
    intent: str


@dataclass
class SessionMemory:
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 6  # keep last N turns to avoid blowing the context window

    def add(self, question: str, answer: str, intent: str):
        self.turns.append(Turn(question, answer, intent))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_prompt_block(self) -> Optional[str]:
        """Format prior turns for injection into LLM prompts."""
        if not self.turns:
            return None
        lines = ["Prior conversation (for context — do not re-answer these):"]
        for t in self.turns:
            lines.append(f"  Q: {t.question}")
            lines.append(f"  A: {t.answer[:200]}{'...' if len(t.answer) > 200 else ''}")
        return "\n".join(lines)
