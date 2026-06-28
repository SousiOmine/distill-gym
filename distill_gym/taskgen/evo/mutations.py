import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from distill_gym.config.schema import TaskItem
from distill_gym.taskgen.evo.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    task: TaskItem
    strategy: str
    parent_ids: list[str]


class MutationStrategy(ABC):
    """Base class for mutation strategies that produce harder task variants."""

    name: str = "base"

    @abstractmethod
    async def mutate(
        self,
        parents: list[TaskItem],
        concepts: list[str],
        llm: LLMClient,
    ) -> MutationResult | None:
        ...

    def _build_task(self, raw: dict, strategy: str, parent_ids: list[str]) -> TaskItem:
        return TaskItem(
            id=str(raw.get("id") or f"evo_{strategy}_{random.randint(0, 999999):06d}"),
            title=str(raw.get("title") or f"Evolved task ({strategy})"),
            prompt=str(raw["prompt"]),
            test_command=raw.get("test_command"),
        )


class ExtendMutation(MutationStrategy):
    """Single-problem extension: take one parent and make it harder.

    Inspired by UniCode's single-problem extension strategy.
    """

    name = "extend"

    async def mutate(self, parents, concepts, llm):
        if not parents:
            return None
        parent = random.choice(parents)
        system = (
            "You are an expert competitive programming problem setter. "
            "Your goal is to create a HARDER variant of an existing coding task. "
            "The new task must require more sophisticated algorithms or data structures. "
            "Return a JSON object with id, title, prompt, and optional test_command."
        )
        user = (
            f"Original task:\nTitle: {parent.title}\nPrompt: {parent.prompt}\n\n"
            f"Target concepts: {', '.join(concepts) if concepts else 'any'}\n\n"
            "Create a harder variant by one of:\n"
            "- Increasing input size or constraints\n"
            "- Adding edge cases that require special handling\n"
            "- Combining multiple sub-problems into one\n"
            "- Requiring a more efficient algorithmic approach\n\n"
            "The new task must be solvable and have a verifiable solution. "
            "Return only the JSON object."
        )
        try:
            result = await llm.complete_json(system, user)
            if isinstance(result, dict) and "prompt" in result:
                task = self._build_task(result, self.name, [parent.id])
                return MutationResult(task=task, strategy=self.name, parent_ids=[parent.id])
        except Exception as exc:
            logger.warning("ExtendMutation failed: %s", exc)
        return None


class FuseSameTypeMutation(MutationStrategy):
    """Fuse two problems of the same type into a harder combined problem.

    Inspired by UniCode's same-type fusion strategy.
    """

    name = "fuse_same_type"

    async def mutate(self, parents, concepts, llm):
        if len(parents) < 2:
            return None
        p1, p2 = random.sample(parents, 2)
        system = (
            "You are an expert competitive programming problem setter. "
            "Merge two related coding tasks into a single harder task that "
            "requires solving both sub-problems and combining their results. "
            "Return a JSON object with id, title, prompt, and optional test_command."
        )
        user = (
            f"Task A:\nTitle: {p1.title}\nPrompt: {p1.prompt}\n\n"
            f"Task B:\nTitle: {p2.title}\nPrompt: {p2.prompt}\n\n"
            f"Target concepts: {', '.join(concepts) if concepts else 'any'}\n\n"
            "Create a single task that combines both tasks into one harder problem. "
            "The combined task should be more difficult than either alone. "
            "Return only the JSON object."
        )
        try:
            result = await llm.complete_json(system, user)
            if isinstance(result, dict) and "prompt" in result:
                task = self._build_task(result, self.name, [p1.id, p2.id])
                return MutationResult(task=task, strategy=self.name, parent_ids=[p1.id, p2.id])
        except Exception as exc:
            logger.warning("FuseSameTypeMutation failed: %s", exc)
        return None


class FuseCrossTypeMutation(MutationStrategy):
    """Fuse two problems of different types to create a novel hybrid.

    Inspired by UniCode's cross-type fusion strategy.
    """

    name = "fuse_cross_type"

    async def mutate(self, parents, concepts, llm):
        if len(parents) < 2:
            return None
        p1, p2 = random.sample(parents, 2)
        system = (
            "You are an expert competitive programming problem setter. "
            "Create a novel hybrid task by combining two different types of "
            "coding problems. The hybrid should require skills from both domains. "
            "Return a JSON object with id, title, prompt, and optional test_command."
        )
        user = (
            f"Task A (one type):\nTitle: {p1.title}\nPrompt: {p1.prompt}\n\n"
            f"Task B (another type):\nTitle: {p2.title}\nPrompt: {p2.prompt}\n\n"
            f"Target concepts: {', '.join(concepts) if concepts else 'any'}\n\n"
            "Create a hybrid task that requires techniques from both tasks. "
            "The result should be a novel problem that is harder than either parent. "
            "Return only the JSON object."
        )
        try:
            result = await llm.complete_json(system, user)
            if isinstance(result, dict) and "prompt" in result:
                task = self._build_task(result, self.name, [p1.id, p2.id])
                return MutationResult(task=task, strategy=self.name, parent_ids=[p1.id, p2.id])
        except Exception as exc:
            logger.warning("FuseCrossTypeMutation failed: %s", exc)
        return None


class AddConstraintMutation(MutationStrategy):
    """Add additional constraints to make a task harder.

    Inspired by BenchEvolver's constraint-driven evolution.
    """

    name = "add_constraint"

    async def mutate(self, parents, concepts, llm):
        if not parents:
            return None
        parent = random.choice(parents)
        system = (
            "You are an expert competitive programming problem setter. "
            "Make a task harder by adding constraints that rule out naive solutions. "
            "Return a JSON object with id, title, prompt, and optional test_command."
        )
        user = (
            f"Original task:\nTitle: {parent.title}\nPrompt: {parent.prompt}\n\n"
            f"Target concepts: {', '.join(concepts) if concepts else 'any'}\n\n"
            "Add one or more of these difficulty-boosting constraints:\n"
            "- Tighter time/space complexity limits\n"
            "- Additional input/output format requirements\n"
            "- Edge case requirements (empty input, maximum values, etc.)\n"
            "- Multiple queries or batch processing\n"
            "- Interactive or online processing requirements\n\n"
            "The task must remain solvable. Return only the JSON object."
        )
        try:
            result = await llm.complete_json(system, user)
            if isinstance(result, dict) and "prompt" in result:
                task = self._build_task(result, self.name, [parent.id])
                return MutationResult(task=task, strategy=self.name, parent_ids=[parent.id])
        except Exception as exc:
            logger.warning("AddConstraintMutation failed: %s", exc)
        return None


class CombineConceptsMutation(MutationStrategy):
    """Generate a new task from sampled concept combinations.

    Inspired by QueST's concept-graph-guided generation.
    """

    name = "combine_concepts"

    async def mutate(self, parents, concepts, llm):
        if not concepts:
            return None
        # Use parents as few-shot examples
        examples = []
        for p in parents[:3]:
            examples.append(f"- {p.title}: {p.prompt[:200]}")
        examples_text = "\n".join(examples) if examples else "(no examples)"

        system = (
            "You are an expert competitive programming problem setter. "
            "Generate a new coding task that requires the specified concepts. "
            "Return a JSON object with id, title, prompt, and optional test_command."
        )
        user = (
            f"Generate a challenging coding task that combines these concepts:\n"
            f"{', '.join(concepts)}\n\n"
            f"Reference tasks for style:\n{examples_text}\n\n"
            "The task should require multiple of the listed concepts to solve. "
            "It must be solvable and have a verifiable solution. "
            "Return only the JSON object."
        )
        try:
            result = await llm.complete_json(system, user)
            if isinstance(result, dict) and "prompt" in result:
                parent_ids = [p.id for p in parents[:3]]
                task = self._build_task(result, self.name, parent_ids)
                return MutationResult(task=task, strategy=self.name, parent_ids=parent_ids)
        except Exception as exc:
            logger.warning("CombineConceptsMutation failed: %s", exc)
        return None


_STRATEGY_MAP: dict[str, type[MutationStrategy]] = {
    "extend": ExtendMutation,
    "fuse_same_type": FuseSameTypeMutation,
    "fuse_cross_type": FuseCrossTypeMutation,
    "add_constraint": AddConstraintMutation,
    "combine_concepts": CombineConceptsMutation,
}


def get_mutation_strategies(names: list[str]) -> list[MutationStrategy]:
    """Instantiate mutation strategies by name."""
    strategies: list[MutationStrategy] = []
    for name in names:
        cls = _STRATEGY_MAP.get(name)
        if cls:
            strategies.append(cls())
    if not strategies:
        strategies = [ExtendMutation()]
    return strategies
