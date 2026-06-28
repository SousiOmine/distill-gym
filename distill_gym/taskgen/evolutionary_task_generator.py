import os
import logging

from distill_gym.config.schema import TaskGenConfig, TaskItem, ProviderConfig
from distill_gym.taskgen.base import TaskGenerator
from distill_gym.taskgen.evo.evolver import Evolver
from distill_gym.taskgen.evo.llm import LLMClient
from distill_gym.registry.taskgen_registry import TaskGenRegistry

logger = logging.getLogger(__name__)


@TaskGenRegistry.register("evolutionary")
class EvolutionaryTaskGenerator(TaskGenerator):
    """Task generator that uses evolutionary algorithms to produce high-difficulty tasks.

    This generator integrates techniques from multiple research papers on
    automatic high-difficulty task generation:

    - QueST (arXiv:2510.17715): concept-graph-guided generation with random walks
    - InfoSynth (arXiv:2601.00575): genetic-algorithm-based synthesis with code feedback
    - UniCode (arXiv:2510.17868): multi-strategy mutation (extension, fusion, cross-fusion)
    - ACES (NeurIPS 2024): Quality-Diversity archive with empirical difficulty estimation
    - BenchEvolver (arXiv:2606.01286): solution-centric evolution with constraint addition

    Configuration is via the ``evolutionary`` field of ``TaskGenConfig``.
    """

    def __init__(
        self,
        config: TaskGenConfig,
        provider_config: ProviderConfig | None = None,
        **kwargs,
    ):
        self.config = config
        self._provider_config = provider_config
        self._seed_tasks = kwargs.get("seed_tasks")

    async def generate(self, count: int, run_id: str = "") -> list[TaskItem]:
        # If static tasks are configured, return them
        if self.config.tasks:
            return self.config.tasks[:count]

        evo_config = self.config.evolutionary

        # If no provider is configured, return seed tasks or fallback
        if not self._provider_config or not self._provider_config.base_url:
            logger.warning("EvolutionaryTaskGenerator: no provider configured, returning seed/fallback tasks")
            return self._fallback(evo_config, count)

        api_key = os.environ.get(self._provider_config.api_key_env, "")
        if not api_key:
            logger.warning("EvolutionaryTaskGenerator: no API key in env %s", self._provider_config.api_key_env)
            return self._fallback(evo_config, count)

        solver_model = evo_config.solver_model or self._provider_config.model
        llm = LLMClient(
            base_url=self._provider_config.base_url,
            api_key=api_key,
            model=self._provider_config.model,
            temperature=evo_config.temperature,
            extra_body=self._provider_config.extra_body,
        )

        seed_tasks = self._seed_tasks or evo_config.seed_tasks
        evolver = Evolver(evo_config, llm, seed_tasks=seed_tasks)
        tasks = await evolver.evolve(count)

        if tasks:
            return tasks

        return self._fallback(evo_config, count)

    def _fallback(self, evo_config, count: int) -> list[TaskItem]:
        """Return seed tasks or generate simple fallback tasks."""
        if evo_config.seed_tasks:
            return evo_config.seed_tasks[:count]

        tasks = []
        for i in range(count):
            tasks.append(TaskItem(
                id=f"evo_fallback_{i:03d}",
                title=f"Evolutionary fallback task {i + 1}",
                prompt=(
                    "Implement a function that solves a non-trivial algorithmic problem. "
                    "Consider edge cases, optimize for time and space complexity, "
                    "and write comprehensive tests."
                ),
            ))
        return tasks
