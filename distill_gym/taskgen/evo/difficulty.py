import logging
from dataclasses import dataclass

from distill_gym.config.schema import TaskItem
from distill_gym.taskgen.evo.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class DifficultyResult:
    """Estimated difficulty of a task.

    score: 0.0 (trivial) to 1.0 (unsolvable by the solver model).
    solver_attempts: how many solve attempts were made.
    solver_successes: how many attempts produced a correct-looking solution.
    """

    score: float
    solver_attempts: int
    solver_successes: int


class DifficultyEstimator:
    """Estimate task difficulty empirically by attempting to solve it.

    Following ACES (NeurIPS 2024), difficulty is defined as:
        difficulty = 1 - success_rate
    where success_rate = solver_successes / solver_attempts.
    """

    def __init__(self, llm: LLMClient, attempts: int = 3):
        self.llm = llm
        self.attempts = max(1, attempts)

    async def estimate(self, task: TaskItem) -> DifficultyResult:
        successes = 0
        for _ in range(self.attempts):
            if await self._try_solve(task):
                successes += 1
        success_rate = successes / self.attempts
        score = 1.0 - success_rate
        return DifficultyResult(
            score=score,
            solver_attempts=self.attempts,
            solver_successes=successes,
        )

    async def _try_solve(self, task: TaskItem) -> bool:
        """Attempt to solve the task. Returns True if the solver produces a plausible solution."""
        system = (
            "You are a competitive programming solver. "
            "Given a coding task, produce a solution. "
            "If the task is too vague or unsolvable, respond with 'UNSOLVABLE'. "
            "Otherwise, provide a brief solution approach and indicate confidence "
            "as 'CONFIDENT' or 'UNCERTAIN'."
        )
        user = (
            f"Task:\nTitle: {task.title}\nPrompt: {task.prompt}\n\n"
            "Provide your solution approach and confidence level."
        )
        try:
            content = await self.llm.complete(system, user)
            # Heuristic: the solver is "successful" if it produces a solution
            # and is confident, and does not declare the task unsolvable.
            content_lower = content.lower()
            if "unsolvable" in content_lower:
                return False
            if "confident" in content_lower:
                return True
            # If uncertain, count as partial success
            if "uncertain" in content_lower:
                return False
            return True
        except Exception as exc:
            logger.warning("Solver attempt failed for %s: %s", task.id, exc)
            return False


def estimate_proxy_difficulty(task: TaskItem) -> float:
    """Quick heuristic difficulty estimate without calling an LLM.

    Uses surface features of the task prompt as a proxy for difficulty.
    Higher values indicate harder tasks.
    """
    prompt = task.prompt.lower()
    score = 0.0

    # Complexity keywords
    hard_keywords = [
        "dynamic programming", "graph", "tree", "segment tree", "binary search",
        " bitmask", "dp ", "memoiz", "recursion", "backtrack", "greedy",
        "optim", "efficient", "linear time", "logarithm", "combinator",
        "modular", "prime", "number theory", "geometry", "probability",
        "concurrent", "parallel", "lock", "mutex", "distributed",
        "parser", "compiler", "interpreter", "state machine",
        "minimum", "maximum", "shortest path", "longest", "optimal",
        "constraint", "edge case", "corner case", "boundary",
    ]
    for kw in hard_keywords:
        if kw in prompt:
            score += 0.05

    # Length-based difficulty (longer prompts tend to be more complex)
    if len(task.prompt) > 500:
        score += 0.1
    if len(task.prompt) > 1000:
        score += 0.1

    # Multiple requirements
    req_markers = prompt.count("must") + prompt.count("should") + prompt.count("require")
    score += min(req_markers * 0.03, 0.15)

    return min(score, 1.0)
