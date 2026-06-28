import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from distill_gym.config.schema import TaskItem, TaskGenConfig, EvolutionaryConfig, ProviderConfig
from distill_gym.taskgen.evo.llm import LLMClient, _parse_json_content
from distill_gym.taskgen.evo.concept_graph import (
    Concept,
    ConceptGraph,
    extract_concepts,
    build_concept_graph,
    sample_concept_combination,
)
from distill_gym.taskgen.evo.mutations import (
    MutationResult,
    MutationStrategy,
    ExtendMutation,
    FuseSameTypeMutation,
    CombineConceptsMutation,
    get_mutation_strategies,
)
from distill_gym.taskgen.evo.difficulty import DifficultyResult, DifficultyEstimator, estimate_proxy_difficulty
from distill_gym.taskgen.evo.archive import ArchiveEntry, QualityDiversityArchive
from distill_gym.taskgen.evo.evolver import Evolver
from distill_gym.taskgen.evolutionary_task_generator import EvolutionaryTaskGenerator
from distill_gym.registry.taskgen_registry import TaskGenRegistry


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self):
        llm = LLMClient(base_url="http://test.local/v1", api_key="test-key", model="test-model")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello, world!"}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("distill_gym.taskgen.evo.llm.httpx.AsyncClient", return_value=mock_client):
            result = await llm.complete("sys prompt", "user prompt")

        assert result == "Hello, world!"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_json_parses_array(self):
        llm = LLMClient(base_url="http://test.local/v1", api_key="test-key", model="test-model")
        with patch.object(llm, "complete") as mock_complete:
            mock_complete.return_value = '```json\n[{"name": "dp", "kind": "knowledge_point"}]\n```'
            result = await llm.complete_json("sys", "user")

        assert result == [{"name": "dp", "kind": "knowledge_point"}]

    @pytest.mark.asyncio
    async def test_complete_json_parses_object(self):
        llm = LLMClient(base_url="http://test.local/v1", api_key="test-key", model="test-model")
        with patch.object(llm, "complete") as mock_complete:
            mock_complete.return_value = '{"id": "t1", "title": "Test", "prompt": "Do it"}'
            result = await llm.complete_json("sys", "user")

        assert result == {"id": "t1", "title": "Test", "prompt": "Do it"}

    def test_parse_json_content_plain_array(self):
        result = _parse_json_content('[{"name": "dp", "kind": "topic"}]')
        assert result == [{"name": "dp", "kind": "topic"}]

    def test_parse_json_content_markdown(self):
        result = _parse_json_content('```json\n[{"name": "dp"}]\n```')
        assert result == [{"name": "dp"}]

    def test_parse_json_content_with_surrounding_text(self):
        content = 'Here is the result:\n```json\n[{"name": "dp"}]\n```\nEnd'
        result = _parse_json_content(content)
        assert result == [{"name": "dp"}]


class TestConceptGraph:
    def test_add_concept_and_co_occurrence(self):
        graph = ConceptGraph()
        graph.add_concept("dp")
        graph.add_concept("graph")
        graph.add_co_occurrence("dp", "graph")

        assert "dp" in graph.nodes
        assert "graph" in graph.nodes
        assert ("dp", "graph") in graph.edges or ("graph", "dp") in graph.edges
        assert graph.edges[("dp", "graph")] == 1.0

    def test_add_co_occurrence_same_concept_ignored(self):
        graph = ConceptGraph()
        graph.add_co_occurrence("dp", "dp")
        assert ("dp", "dp") not in graph.edges

    def test_random_walk_returns_visited_nodes(self):
        graph = ConceptGraph()
        graph.add_co_occurrence("dp", "graph")
        graph.add_co_occurrence("graph", "tree")
        graph.add_co_occurrence("tree", "dp")

        visited = graph.random_walk(start="dp", steps=3)
        assert len(visited) == 4
        for node in visited:
            assert node in graph.nodes

    def test_random_walk_empty_graph(self):
        graph = ConceptGraph()
        visited = graph.random_walk(steps=3)
        assert visited == []

    def test_build_concept_graph(self):
        tasks = [
            TaskItem(id="t1", title="Task1", prompt="Do DP"),
            TaskItem(id="t2", title="Task2", prompt="Do graph"),
        ]
        task_concepts = {
            "t1": [
                Concept(name="dp", kind="knowledge_point"),
                Concept(name="memoization", kind="knowledge_point"),
            ],
            "t2": [
                Concept(name="graph", kind="knowledge_point"),
                Concept(name="dp", kind="knowledge_point"),
            ],
        }
        graph = build_concept_graph(tasks, task_concepts)
        assert "dp" in graph.nodes
        assert "memoization" in graph.nodes
        assert "graph" in graph.nodes
        assert ("dp", "memoization") in graph.edges
        assert ("dp", "graph") in graph.edges

    @pytest.mark.asyncio
    async def test_extract_concepts_success(self):
        task = TaskItem(id="t1", title="Task", prompt="Solve a DP problem")
        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm.complete_json = AsyncMock(
            return_value=[
                {"name": "dynamic programming", "kind": "topic"},
                {"name": "memoization", "kind": "knowledge_point"},
            ]
        )

        concepts = await extract_concepts(mock_llm, task)

        assert len(concepts) == 2
        assert concepts[0].name == "dynamic programming"
        assert concepts[0].kind == "topic"
        assert concepts[1].name == "memoization"
        assert concepts[1].kind == "knowledge_point"

    @pytest.mark.asyncio
    async def test_extract_concepts_failure(self):
        task = TaskItem(id="t1", title="Task", prompt="Solve a DP problem")
        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm.complete_json = AsyncMock(side_effect=Exception("API error"))

        concepts = await extract_concepts(mock_llm, task)

        assert concepts == []


class TestMutations:
    def test_get_mutation_strategies_all(self):
        names = ["extend", "fuse_same_type", "fuse_cross_type", "add_constraint", "combine_concepts"]
        strategies = get_mutation_strategies(names)
        assert len(strategies) == 5
        result_names = {s.name for s in strategies}
        assert result_names == set(names)

    def test_get_mutation_strategies_empty_fallback(self):
        strategies = get_mutation_strategies([])
        assert len(strategies) == 1
        assert strategies[0].name == "extend"

    def test_get_mutation_strategies_unknown_ignored(self):
        strategies = get_mutation_strategies(["unknown_strat", "also_unknown"])
        assert len(strategies) == 1
        assert strategies[0].name == "extend"

    @pytest.mark.asyncio
    async def test_extend_mutation_success(self):
        parents = [TaskItem(id="t1", title="Task1", prompt="Do something")]
        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm.complete_json = AsyncMock(
            return_value={
                "id": "evo_extend_001",
                "title": "Harder Task",
                "prompt": "Solve a harder problem",
            }
        )

        mutation = ExtendMutation()
        result = await mutation.mutate(parents, ["dp", "graph"], mock_llm)

        assert result is not None
        assert result.strategy == "extend"
        assert result.parent_ids == ["t1"]
        assert result.task.title == "Harder Task"
        assert result.task.prompt == "Solve a harder problem"

    @pytest.mark.asyncio
    async def test_extend_mutation_no_parents(self):
        mock_llm = AsyncMock(spec=LLMClient)
        mutation = ExtendMutation()
        result = await mutation.mutate([], ["dp"], mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_fuse_same_type_mutation_success(self):
        parents = [
            TaskItem(id="t1", title="Task1", prompt="Do sorting"),
            TaskItem(id="t2", title="Task2", prompt="Do searching"),
        ]
        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm.complete_json = AsyncMock(
            return_value={
                "id": "evo_fuse_001",
                "title": "Combined Task",
                "prompt": "Sort and search together",
            }
        )

        mutation = FuseSameTypeMutation()
        result = await mutation.mutate(parents, ["dp"], mock_llm)

        assert result is not None
        assert result.strategy == "fuse_same_type"
        assert len(result.parent_ids) == 2
        assert result.task.prompt == "Sort and search together"

    @pytest.mark.asyncio
    async def test_fuse_same_type_mutation_single_parent(self):
        parents = [TaskItem(id="t1", title="Task1", prompt="Do it")]
        mock_llm = AsyncMock(spec=LLMClient)
        mutation = FuseSameTypeMutation()
        result = await mutation.mutate(parents, ["dp"], mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_combine_concepts_mutation_no_concepts(self):
        parents = [TaskItem(id="t1", title="Task1", prompt="Do it")]
        mock_llm = AsyncMock(spec=LLMClient)
        mutation = CombineConceptsMutation()
        result = await mutation.mutate(parents, [], mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_combine_concepts_mutation_success(self):
        parents = [TaskItem(id="t1", title="Task1", prompt="Do it")]
        concepts = ["dp", "graph"]
        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm.complete_json = AsyncMock(
            return_value={
                "id": "evo_combine_001",
                "title": "Combined Task",
                "prompt": "Use DP and graph concepts",
            }
        )

        mutation = CombineConceptsMutation()
        result = await mutation.mutate(parents, concepts, mock_llm)

        assert result is not None
        assert result.strategy == "combine_concepts"
        assert result.task.prompt == "Use DP and graph concepts"


class TestDifficulty:
    def test_estimate_proxy_difficulty_keywords(self):
        task = TaskItem(
            id="t1",
            title="Task",
            prompt=(
                "Design a dynamic programming algorithm for the minimum path sum "
                "problem on a graph. Must handle edge cases and optimize for "
                "both time and space complexity."
            ),
        )
        score = estimate_proxy_difficulty(task)
        assert score > 0.2

    def test_estimate_proxy_difficulty_long_prompt(self):
        long_prompt = "Implement a function that does X. " * 50
        task = TaskItem(id="t1", title="Long Task", prompt=long_prompt)
        score = estimate_proxy_difficulty(task)
        assert score >= 0.2

    def test_estimate_proxy_difficulty_simple(self):
        task = TaskItem(id="t1", title="Simple", prompt="print hello world")
        score = estimate_proxy_difficulty(task)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_difficulty_estimator_all_solved(self):
        mock_llm = AsyncMock(spec=LLMClient)
        estimator = DifficultyEstimator(mock_llm, attempts=3)
        estimator._try_solve = AsyncMock(return_value=True)

        task = TaskItem(id="t1", title="Task", prompt="Simple task")
        result = await estimator.estimate(task)

        assert result.score == 0.0
        assert result.solver_successes == 3
        assert result.solver_attempts == 3

    @pytest.mark.asyncio
    async def test_difficulty_estimator_none_solved(self):
        mock_llm = AsyncMock(spec=LLMClient)
        estimator = DifficultyEstimator(mock_llm, attempts=3)
        estimator._try_solve = AsyncMock(return_value=False)

        task = TaskItem(id="t1", title="Task", prompt="Simple task")
        result = await estimator.estimate(task)

        assert result.score == 1.0
        assert result.solver_successes == 0
        assert result.solver_attempts == 3

    @pytest.mark.asyncio
    async def test_difficulty_estimator_partial(self):
        mock_llm = AsyncMock(spec=LLMClient)
        estimator = DifficultyEstimator(mock_llm, attempts=3)
        estimator._try_solve = AsyncMock(side_effect=[True, False, False])

        task = TaskItem(id="t1", title="Task", prompt="Simple task")
        result = await estimator.estimate(task)

        assert result.score == pytest.approx(0.667, abs=0.001)
        assert result.solver_successes == 1
        assert result.solver_attempts == 3


class TestArchive:
    def test_add_first_entry(self):
        archive = QualityDiversityArchive(capacity=10)
        entry = ArchiveEntry(
            task=TaskItem(id="t1", title="Task", prompt="Do it"),
            concepts=["dp"],
            difficulty=0.5,
            generation=1,
        )
        accepted = archive.add(entry)
        assert accepted is True
        assert archive.size == 1

    def test_add_higher_difficulty_replaces(self):
        archive = QualityDiversityArchive(capacity=10)
        entry1 = ArchiveEntry(
            task=TaskItem(id="t1", title="Task", prompt="Do it"),
            concepts=["dp"],
            difficulty=0.3,
            generation=1,
        )
        entry2 = ArchiveEntry(
            task=TaskItem(id="t2", title="Harder", prompt="Hard task"),
            concepts=["dp"],
            difficulty=0.7,
            generation=2,
        )

        assert archive.add(entry1) is True
        assert archive.size == 1
        assert archive.add(entry2) is True
        assert archive.size == 1
        assert archive.get_top(1)[0].difficulty == 0.7

    def test_add_lower_difficulty_rejected(self):
        archive = QualityDiversityArchive(capacity=10)
        entry1 = ArchiveEntry(
            task=TaskItem(id="t1", title="Hard", prompt="Hard task"),
            concepts=["dp"],
            difficulty=0.7,
            generation=1,
        )
        entry2 = ArchiveEntry(
            task=TaskItem(id="t2", title="Easy", prompt="Easy task"),
            concepts=["dp"],
            difficulty=0.3,
            generation=2,
        )

        assert archive.add(entry1) is True
        assert archive.add(entry2) is False
        assert archive.size == 1
        assert archive.get_top(1)[0].difficulty == 0.7

    def test_tournament_selection(self):
        archive = QualityDiversityArchive(capacity=10)
        archive.add(ArchiveEntry(
            task=TaskItem(id="t1", title="Task1", prompt="P1"),
            concepts=["math"],
            difficulty=0.3,
            generation=1,
        ))
        archive.add(ArchiveEntry(
            task=TaskItem(id="t2", title="Task2", prompt="P2"),
            concepts=["algo"],
            difficulty=0.7,
            generation=1,
        ))
        archive.add(ArchiveEntry(
            task=TaskItem(id="t3", title="Task3", prompt="P3"),
            concepts=["system"],
            difficulty=0.5,
            generation=1,
        ))

        selected = archive.select_parent(tournament_size=10)
        assert selected is not None
        assert selected.difficulty == 0.7

    def test_tournament_selection_empty(self):
        archive = QualityDiversityArchive(capacity=10)
        selected = archive.select_parent(tournament_size=3)
        assert selected is None

    def test_get_top(self):
        archive = QualityDiversityArchive(capacity=10)
        for i, diff in enumerate([0.1, 0.9, 0.5, 0.7, 0.3]):
            archive.add(ArchiveEntry(
                task=TaskItem(id=f"t{i}", title=f"Task{i}", prompt=f"P{i}"),
                concepts=[f"concept_{i}"],
                difficulty=diff,
                generation=1,
            ))

        top3 = archive.get_top(3)
        assert len(top3) == 3
        assert top3[0].difficulty == 0.9
        assert top3[1].difficulty == 0.7
        assert top3[2].difficulty == 0.5

    def test_capacity_enforcement(self):
        archive = QualityDiversityArchive(capacity=2)
        archive.add(ArchiveEntry(
            task=TaskItem(id="t1", title="Task1", prompt="P1"),
            concepts=["math"],
            difficulty=0.1,
            generation=1,
        ))
        archive.add(ArchiveEntry(
            task=TaskItem(id="t2", title="Task2", prompt="P2"),
            concepts=["algo"],
            difficulty=0.5,
            generation=1,
        ))
        archive.add(ArchiveEntry(
            task=TaskItem(id="t3", title="Task3", prompt="P3"),
            concepts=["system"],
            difficulty=0.9,
            generation=1,
        ))

        assert archive.size == 2
        tasks = archive.get_all()
        diffs = sorted(e.difficulty for e in tasks)
        assert diffs == [0.5, 0.9]


class TestEvolver:
    @pytest.mark.asyncio
    async def test_evolve_returns_tasks(self):
        evo_config = EvolutionaryConfig(
            seed_tasks=[TaskItem(id="s1", title="Seed", prompt="Do something")],
            max_generations=2,
            population_size=2,
            difficulty_min=0.0,
            solver_attempts=1,
        )

        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm.complete = AsyncMock()
        mock_llm.complete_json = AsyncMock()

        mock_strategy = AsyncMock(spec=MutationStrategy)
        mock_strategy.name = "extend"
        mock_strategy.mutate = AsyncMock(
            return_value=MutationResult(
                task=TaskItem(id="evo_000001", title="Evolved", prompt="Hard task"),
                strategy="extend",
                parent_ids=["s1"],
            )
        )

        with (
            patch("distill_gym.taskgen.evo.evolver.extract_concepts") as mock_extract,
            patch("distill_gym.taskgen.evo.evolver.sample_concept_combination") as mock_sample,
        ):
            mock_extract.return_value = [Concept(name="dp", kind="knowledge_point")]
            mock_sample.return_value = ["dp", "graph"]

            evolver = Evolver(evo_config, mock_llm)
            evolver.strategies = [mock_strategy]
            evolver._difficulty_estimator.estimate = AsyncMock(
                return_value=DifficultyResult(score=0.5, solver_attempts=1, solver_successes=0)
            )

            tasks = await evolver.evolve(count=2)

        assert len(tasks) == 2
        assert all(isinstance(t, TaskItem) for t in tasks)
        assert tasks[0].title == "Evolved" or tasks[0].id == "s1"


class TestEvolutionaryTaskGenerator:
    def test_registered(self):
        gen_class = TaskGenRegistry.get("evolutionary")
        assert gen_class is EvolutionaryTaskGenerator

    @pytest.mark.asyncio
    async def test_static_tasks_from_config(self):
        config = TaskGenConfig(
            type="evolutionary",
            tasks=[
                TaskItem(id="t1", title="Task1", prompt="Do 1"),
                TaskItem(id="t2", title="Task2", prompt="Do 2"),
            ],
        )
        gen = EvolutionaryTaskGenerator(config)
        tasks = await gen.generate(2)

        assert len(tasks) == 2
        assert tasks[0].id == "t1"
        assert tasks[1].id == "t2"

    @pytest.mark.asyncio
    async def test_fallback_without_provider(self):
        config = TaskGenConfig(type="evolutionary")
        gen = EvolutionaryTaskGenerator(config, provider_config=None)
        tasks = await gen.generate(2)

        assert len(tasks) == 2
        assert "fallback" in tasks[0].id

    @pytest.mark.asyncio
    async def test_fallback_without_api_key(self):
        config = TaskGenConfig(type="evolutionary")
        provider_config = ProviderConfig(
            base_url="http://test.local/v1",
            api_key_env="MISSING_ENV_VAR",
            model="test-model",
        )
        gen = EvolutionaryTaskGenerator(config, provider_config=provider_config)
        tasks = await gen.generate(2)

        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_fallback_with_seed_tasks(self):
        evo_config = EvolutionaryConfig(
            seed_tasks=[TaskItem(id="s1", title="Seed", prompt="Do it")],
        )
        config = TaskGenConfig(type="evolutionary", evolutionary=evo_config)
        gen = EvolutionaryTaskGenerator(config, provider_config=None)
        tasks = await gen.generate(2)

        assert len(tasks) == 1
        assert tasks[0].id == "s1"

    @pytest.mark.asyncio
    async def test_generate_calls_evolver(self):
        config = TaskGenConfig(type="evolutionary")
        provider_config = ProviderConfig(
            base_url="http://test.local/v1",
            api_key_env="TEST_EVOLVER_KEY",
            model="test-model",
        )
        gen = EvolutionaryTaskGenerator(config, provider_config=provider_config)

        os.environ["TEST_EVOLVER_KEY"] = "test-key"
        try:
            with patch(
                "distill_gym.taskgen.evolutionary_task_generator.Evolver",
            ) as MockEvolver:
                mock_evolver = AsyncMock(spec=Evolver)
                mock_evolver.evolve = AsyncMock(
                    return_value=[
                        TaskItem(id="evo_1", title="Evolved", prompt="Hard task 1"),
                    ]
                )
                MockEvolver.return_value = mock_evolver

                tasks = await gen.generate(1)

                assert len(tasks) == 1
                assert tasks[0].id == "evo_1"
                MockEvolver.assert_called_once()
        finally:
            del os.environ["TEST_EVOLVER_KEY"]
