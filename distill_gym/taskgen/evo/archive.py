import random
import logging
from dataclasses import dataclass, field

from distill_gym.config.schema import TaskItem

logger = logging.getLogger(__name__)


@dataclass
class ArchiveEntry:
    task: TaskItem
    concepts: list[str]
    difficulty: float
    generation: int
    strategy: str = ""
    parent_ids: list[str] = field(default_factory=list)


class QualityDiversityArchive:
    """Quality-Diversity archive for evolutionary task generation.

    Inspired by ACES (NeurIPS 2024), this archive maps concept-combination
    niches to the best (highest-difficulty) task in each niche. This promotes
    both diversity (across concept space) and quality (difficulty).

    Each niche is defined by a frozenset of concept names. When a new entry
    arrives, it is accepted only if its niche is empty or its difficulty
    exceeds the current niche champion.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._entries: list[ArchiveEntry] = []
        self._niche_map: dict[frozenset, ArchiveEntry] = {}

    def add(self, entry: ArchiveEntry) -> bool:
        """Try to add an entry. Returns True if accepted into the archive."""
        niche = frozenset(entry.concepts) if entry.concepts else frozenset({"_default_"})
        existing = self._niche_map.get(niche)
        if existing is None or entry.difficulty > existing.difficulty:
            if existing is not None:
                self._entries.remove(existing)
            self._niche_map[niche] = entry
            self._entries.append(entry)
            self._enforce_capacity()
            return True
        return False

    def _enforce_capacity(self) -> None:
        """If over capacity, remove lowest-difficulty entries."""
        if len(self._entries) <= self.capacity:
            return
        self._entries.sort(key=lambda e: e.difficulty, reverse=True)
        removed = self._entries[self.capacity:]
        self._entries = self._entries[: self.capacity]
        for entry in removed:
            niche = frozenset(entry.concepts) if entry.concepts else frozenset({"_default_"})
            if self._niche_map.get(niche) is entry:
                del self._niche_map[niche]

    def select_parent(self, tournament_size: int = 3) -> ArchiveEntry | None:
        """Tournament selection: pick the highest-difficulty among random samples."""
        if not self._entries:
            return None
        k = min(tournament_size, len(self._entries))
        candidates = random.sample(self._entries, k)
        return max(candidates, key=lambda e: e.difficulty)

    def select_parents(self, count: int, tournament_size: int = 3) -> list[ArchiveEntry]:
        """Select multiple parents via tournament selection."""
        parents: list[ArchiveEntry] = []
        for _ in range(count):
            parent = self.select_parent(tournament_size)
            if parent is None:
                break
            parents.append(parent)
        return parents

    def get_top(self, n: int) -> list[ArchiveEntry]:
        """Return the top-N entries by difficulty."""
        return sorted(self._entries, key=lambda e: e.difficulty, reverse=True)[:n]

    def get_all(self) -> list[ArchiveEntry]:
        """Return all entries."""
        return list(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)
