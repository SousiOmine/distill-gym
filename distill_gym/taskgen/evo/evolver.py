import asyncio
import logging
import random

from distill_gym.config.schema import TaskItem, EvolutionaryConfig, ProviderConfig
from distill_gym.taskgen.evo.archive import ArchiveEntry, QualityDiversityArchive
from distill_gym.taskgen.evo.concept_graph import (
    Concept,
    ConceptGraph,
    build_concept_graph,
    extract_concepts,
    sample_concept_combination,
)
from distill_gym.taskgen.evo.difficulty import DifficultyEstimator, estimate_proxy_difficulty
from distill_gym.taskgen.evo.llm import LLMClient
from distill_gym.taskgen.evo.mutations import get_mutation_strategies

logger = logging.getLogger(__name__)


class Evolver:
    """Evolutionary task generator that produces high-difficulty coding tasks.

    Integrates techniques from multiple research papers:

    - **Concept graph + random walk sampling** (QueST, arXiv:2510.17715):
      Seed tasks are analyzed to extract programming concepts; a co-occurrence
      graph guides novel concept-combination sampling via weighted random walks.
    - **Mutation strategies** (UniCode, arXiv:2510.17868; BenchEvolver, arXiv:2606.01286):
      Five strategies—single-problem extension, same-type fusion, cross-type
      fusion, constraint addition, and concept-combination generation—produce
      harder task variants from archive parents.
    - **Empirical difficulty estimation** (ACES, NeurIPS 2024; QueST):
      Difficulty is measured as ``1 - solver_success_rate`` by attempting to
      solve each candidate with a solver model.
    - **Quality-Diversity archive** (ACES):
      A niche-based archive maps concept combinations to the highest-difficulty
      task, promoting both diversity and quality.
    """

    def __init__(
        self,
        config: EvolutionaryConfig,
        llm: LLMClient,
        seed_tasks: list[TaskItem] | None = None,
    ):
        self.config = config
        self.llm = llm
        self.seed_tasks = seed_tasks or list(config.seed_tasks)
        self.strategies = get_mutation_strategies(config.mutation_strategies)
        self.archive = QualityDiversityArchive(capacity=config.archive_capacity)
        self.concept_graph = ConceptGraph()
        self._task_concepts: dict[str, list[Concept]] = {}
        self._difficulty_estimator = DifficultyEstimator(
            llm=llm,
            attempts=config.solver_attempts,
        )

    async def _initialize(self) -> None:
        """Extract concepts from seed tasks and build the concept graph."""
        if not self.seed_tasks:
            logger.warning("No seed tasks provided; evolution starts from scratch")
            return

        for task in self.seed_tasks:
            concepts = await extract_concepts(self.llm, task)
            self._task_concepts[task.id] = concepts
            proxy_diff = estimate_proxy_difficulty(task)
            entry = ArchiveEntry(
                task=task,
                concepts=[c.name for c in concepts],
                difficulty=proxy_diff,
                generation=0,
                strategy="seed",
            )
            self.archive.add(entry)

        self.concept_graph = build_concept_graph(self.seed_tasks, self._task_concepts)
        logger.info(
            "Initialized evolver: %d seed tasks, %d concepts, archive size %d",
            len(self.seed_tasks),
            len(self.concept_graph.nodes),
            self.archive.size,
        )

    async def _generate_offspring(
        self,
        generation: int,
    ) -> ArchiveEntry | None:
        """Generate a single offspring via mutation and evaluate its difficulty."""
        parents_entries = self.archive.select_parents(
            count=2,
            tournament_size=self.config.tournament_size,
        )
        parent_tasks = [e.task for e in parents_entries] if parents_entries else list(self.seed_tasks[:3])
        if not parent_tasks:
            return None

        # sample concept combination from the graph
        concepts = await sample_concept_combination(
            self.concept_graph,
            steps=self.config.concept_graph_steps,
        )
        # deduplicate while preserving order
        seen = set()
        concepts = [c for c in concepts if not (c in seen or seen.add(c))]

        strategy = random.choice(self.strategies)
        try:
            result = await strategy.mutate(parent_tasks, concepts, self.llm)
        except Exception as exc:
            logger.warning("Mutation %s failed: %s", strategy.name, exc)
            return None

        if result is None:
            return None

        # estimate difficulty
        try:
            diff_result = await self._difficulty_estimator.estimate(result.task)
            difficulty = diff_result.score
        except Exception as exc:
            logger.warning("Difficulty estimation failed: %s", exc)
            difficulty = estimate_proxy_difficulty(result.task)

        # extract concepts for the new task (reuse sampled concepts as proxy)
        new_concepts = concepts if concepts else [strategy.name]

        return ArchiveEntry(
            task=result.task,
            concepts=new_concepts,
            difficulty=difficulty,
            generation=generation,
            strategy=result.strategy,
            parent_ids=result.parent_ids,
        )

    async def evolve(self, count: int) -> list[TaskItem]:
        """Run the evolutionary loop and return the top-``count`` tasks by difficulty."""
        await self._initialize()

        for gen in range(1, self.config.max_generations + 1):
            if self.archive.size >= count and gen > self.config.max_generations // 2:
                # early termination if we have enough high-difficulty tasks
                top = self.archive.get_top(count)
                if all(e.difficulty >= self.config.difficulty_min for e in top):
                    logger.info("Early termination at generation %d", gen)
                    break

            # generate population for this generation
            offspring_tasks = []
            for _ in range(self.config.population_size):
                entry = await self._generate_offspring(gen)
                if entry is not None:
                    offspring_tasks.append(entry)

            # add offspring to archive (filtered by difficulty threshold)
            accepted = 0
            for entry in offspring_tasks:
                if entry.difficulty >= self.config.difficulty_min:
                    if self.archive.add(entry):
                        accepted += 1

            logger.info(
                "Generation %d: produced %d offspring, accepted %d, archive size %d",
                gen,
                len(offspring_tasks),
                accepted,
                self.archive.size,
            )

            if self.archive.size == 0 and not offspring_tasks:
                logger.warning("No tasks generated in generation %d; stopping", gen)
                break

        top = self.archive.get_top(count)
        tasks = [e.task for e in top]

        # if not enough tasks, fall back to seeds
        if len(tasks) < count:
            for seed in self.seed_tasks:
                if seed not in tasks:
                    tasks.append(seed)
                if len(tasks) >= count:
                    break

        return tasks[:count]
