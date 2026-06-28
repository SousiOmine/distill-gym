import math
import random
import logging
from dataclasses import dataclass, field

from distill_gym.config.schema import TaskItem
from distill_gym.taskgen.evo.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    name: str
    kind: str = "knowledge_point"  # "topic" or "knowledge_point"


@dataclass
class ConceptGraph:
    """Graph of concept co-occurrence, used for random-walk sampling.

    Nodes are concept names; edge weights encode co-occurrence frequency
    following QueST/MathScale: w(u,v) = log(freq(u,v) + epsilon).
    """

    nodes: set[str] = field(default_factory=set)
    edges: dict[tuple[str, str], float] = field(default_factory=dict)
    _adj: dict[str, list[str]] = field(default_factory=dict)
    _epsilon: float = 1e-6

    def add_concept(self, name: str) -> None:
        self.nodes.add(name)
        self._adj.setdefault(name, [])

    def add_co_occurrence(self, c1: str, c2: str, count: int = 1) -> None:
        if c1 == c2:
            return
        key = tuple(sorted((c1, c2)))
        self.edges[key] = self.edges.get(key, 0.0) + count
        self.add_concept(c1)
        self.add_concept(c2)
        if c2 not in self._adj[c1]:
            self._adj[c1].append(c2)
        if c1 not in self._adj[c2]:
            self._adj[c2].append(c1)

    def _weight(self, c1: str, c2: str) -> float:
        key = tuple(sorted((c1, c2)))
        return math.log(self.edges.get(key, 0.0) + self._epsilon)

    def random_walk(self, start: str | None = None, steps: int = 6) -> list[str]:
        """Perform a weighted random walk and return visited concept names."""
        if not self.nodes:
            return []
        if start is None:
            start = random.choice(list(self.nodes))
        if start not in self.nodes:
            start = random.choice(list(self.nodes))

        visited = [start]
        current = start
        for _ in range(steps):
            neighbors = self._adj.get(current, [])
            if not neighbors:
                break
            weights = [self._weight(current, n) for n in neighbors]
            # softmax transition probabilities (Equation 3 in QueST)
            max_w = max(weights)
            exps = [math.exp(w - max_w) for w in weights]
            total = sum(exps)
            probs = [e / total for e in exps]
            current = random.choices(neighbors, weights=probs, k=1)[0]
            visited.append(current)
        return visited


async def extract_concepts(
    llm: LLMClient,
    task: TaskItem,
) -> list[Concept]:
    """Extract concepts (topics and knowledge points) from a seed task via LLM."""
    system = (
        "You are a programming concept extractor. "
        "Given a coding task, extract the key programming concepts, "
        "algorithms, data structures, and techniques it involves. "
        "Return a JSON array of objects, each with 'name' and 'kind' "
        "('topic' for broad areas, 'knowledge_point' for specific concepts)."
    )
    user = (
        f"Task ID: {task.id}\n"
        f"Title: {task.title}\n"
        f"Prompt: {task.prompt}\n\n"
        "Extract 3-8 concepts. Return only the JSON array."
    )
    try:
        result = await llm.complete_json(system, user)
        if isinstance(result, list):
            concepts = []
            for item in result:
                if isinstance(item, dict) and "name" in item:
                    concepts.append(Concept(
                        name=str(item["name"]),
                        kind=str(item.get("kind", "knowledge_point")),
                    ))
            return concepts
    except Exception as exc:
        logger.warning("Concept extraction failed for %s: %s", task.id, exc)
    return []


def build_concept_graph(
    tasks: list[TaskItem],
    task_concepts: dict[str, list[Concept]],
) -> ConceptGraph:
    """Build a concept co-occurrence graph from task-concept mappings."""
    graph = ConceptGraph()
    for task in tasks:
        concepts = task_concepts.get(task.id, [])
        names = [c.name for c in concepts]
        for c in concepts:
            graph.add_concept(c.name)
        # add co-occurrence edges between all pairs in the same task
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                graph.add_co_occurrence(a, b)
    return graph


async def sample_concept_combination(
    graph: ConceptGraph,
    steps: int = 6,
) -> list[str]:
    """Sample a concept combination via random walk on the graph."""
    return graph.random_walk(steps=steps)
