"""
The Nexus Core - Research Swarm

Lightweight, zero-LLM research persona system with continuous NEAT-style
evolutionary pressure.

Each ResearchPersona reformulates queries using a rule-based strategy (no LLM
required per searcher).  All active personas search the knowledge base in
a single pass; their results are merged and deduplicated before a normal LLM
synthesis call.

Evolution is continuous:
  - Every `competition_interval` searches the lowest-fitness active persona
    with enough samples competes against a mutated challenger bred from the
    current leader.
  - The loser is eliminated; the challenger takes its slot.
  - No persona is ever permanently immune — even repeated winners face
    challengers over time.

Persistence
-----------
  nexus_data/swarm/personas.jsonl   — all personas (active + eliminated)
  nexus_data/swarm/active.json      — list of currently active persona IDs
"""

import json
import random
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from nexus_core_config import config as _config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard-coded term vocabularies per strategy type.
# These are used both for default initialisation and when mutating.
_STRATEGY_VOCAB: Dict[str, Dict[str, List[str]]] = {
    "technical": {
        "prefix": [
            "technical details about", "implementation of",
            "specification for", "how does", "architecture of",
            "internals of", "mechanism behind",
        ],
        "suffix": [
            "algorithm", "implementation", "specification", "protocol",
            "technical overview", "data structure", "complexity",
        ],
    },
    "broad": {
        "prefix": [
            "overview of", "introduction to", "background on",
            "general information about", "summary of",
            "context for", "what is",
        ],
        "suffix": [
            "overview", "background", "introduction", "context",
            "general concept", "related topics", "broad perspective",
        ],
    },
    "skeptical": {
        "prefix": [
            "limitations of", "problems with", "criticism of",
            "failure cases for", "risks of", "drawbacks of",
            "challenges with",
        ],
        "suffix": [
            "limitations", "problems", "criticisms", "failure modes",
            "risks", "downsides", "known issues",
        ],
    },
    "practical": {
        "prefix": [
            "how to use", "practical guide to", "step-by-step",
            "examples of", "tutorial for", "how to implement",
            "best practices for",
        ],
        "suffix": [
            "examples", "tutorial", "how-to", "use cases",
            "practical application", "workflow", "guide",
        ],
    },
    "historical": {
        "prefix": [
            "history of", "origin of", "evolution of",
            "development of", "background history of",
            "timeline of", "when was",
        ],
        "suffix": [
            "history", "origins", "evolution", "development timeline",
            "historical context", "background", "past",
        ],
    },
    "definitions": {
        "prefix": [
            "definition of", "what is", "meaning of",
            "explain", "describe", "concept of",
            "terminology for",
        ],
        "suffix": [
            "definition", "meaning", "explanation", "description",
            "concept", "terminology", "glossary",
        ],
    },
    "comparative": {
        "prefix": [
            "comparison of", "difference between", "versus",
            "how does ... compare to", "tradeoffs between",
            "alternatives to", "contrast between",
        ],
        "suffix": [
            "comparison", "versus", "tradeoffs", "alternatives",
            "differences", "similarities", "contrast",
        ],
    },
}

_STRATEGIES = list(_STRATEGY_VOCAB.keys())

# Default seed personas — one per strategy except comparative (seeded later)
_SEED_PERSONAS = [
    ("Technical Analyst",    "technical"),
    ("Broad Context Seeker", "broad"),
    ("Critical Examiner",    "skeptical"),
    ("Practical Applier",    "practical"),
    ("Historical Researcher","historical"),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:8]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick(vocab_list: List[str]) -> str:
    return random.choice(vocab_list)


# ---------------------------------------------------------------------------
# ResearchPersona dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResearchPersona:
    """
    An evolvable query-reformulation strategy.

    Genes
    -----
    strategy    — one of the 7 strategy types
    prefix      — phrase prepended to the raw query
    suffix      — phrase appended to the reformulated query

    Fitness tracking
    ----------------
    fitness         — running average of per-search utility in [0.0, 1.0].
    search_count    — total searches this persona has participated in.
    hit_count       — searches where this persona contributed at least one
                      chunk that made it into the final merged result.
    active          — False once this persona has been eliminated.
    """
    persona_id:  str
    name:        str
    strategy:    str
    prefix:      str
    suffix:      str
    fitness:     float
    search_count: int
    hit_count:   int
    active:      bool
    generation:  int
    parent_id:   Optional[str]
    created_at:  str

    # ----- Computed helpers -------------------------------------------------

    def reformulate(self, query: str) -> str:
        """Produce the search query this persona would issue."""
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        parts.append(query)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)

    def record_search(self, hit: bool) -> None:
        """Update fitness after one search."""
        self.search_count += 1
        if hit:
            self.hit_count += 1
        n = self.search_count
        new_sample = 1.0 if hit else 0.0
        self.fitness = round((self.fitness * (n - 1) + new_sample) / n, 4)

    # ----- Serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id":   self.persona_id,
            "name":         self.name,
            "strategy":     self.strategy,
            "prefix":       self.prefix,
            "suffix":       self.suffix,
            "fitness":      self.fitness,
            "search_count": self.search_count,
            "hit_count":    self.hit_count,
            "active":       self.active,
            "generation":   self.generation,
            "parent_id":    self.parent_id,
            "created_at":   self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ResearchPersona":
        return ResearchPersona(
            persona_id=d["persona_id"],
            name=d["name"],
            strategy=d["strategy"],
            prefix=d.get("prefix", ""),
            suffix=d.get("suffix", ""),
            fitness=d.get("fitness", 0.5),
            search_count=d.get("search_count", 0),
            hit_count=d.get("hit_count", 0),
            active=d.get("active", True),
            generation=d.get("generation", 0),
            parent_id=d.get("parent_id"),
            created_at=d.get("created_at", _now_iso()),
        )


# ---------------------------------------------------------------------------
# Mutation / breeding
# ---------------------------------------------------------------------------

def _seed_persona(name: str, strategy: str, generation: int = 0,
                  parent_id: Optional[str] = None) -> ResearchPersona:
    """Create a new persona with default vocab terms for its strategy."""
    vocab = _STRATEGY_VOCAB[strategy]
    return ResearchPersona(
        persona_id=_new_id(),
        name=name,
        strategy=strategy,
        prefix=_pick(vocab["prefix"]),
        suffix=_pick(vocab["suffix"]),
        fitness=0.5,          # neutral starting fitness
        search_count=0,
        hit_count=0,
        active=True,
        generation=generation,
        parent_id=parent_id,
        created_at=_now_iso(),
    )


def mutate_persona(parent: ResearchPersona) -> ResearchPersona:
    """
    Produce a child persona from *parent*.

    Mutations (any combination):
      - Change the strategy type entirely (20 % chance)
      - Pick a new prefix from the (possibly new) strategy vocab (50 %)
      - Pick a new suffix from the (possibly new) strategy vocab (50 %)
    At least one of prefix or suffix always changes.
    """
    strategy = parent.strategy
    if random.random() < 0.20:
        strategy = random.choice([s for s in _STRATEGIES if s != strategy])

    vocab = _STRATEGY_VOCAB[strategy]

    mutate_prefix = random.random() < 0.50
    mutate_suffix = random.random() < 0.50
    if not mutate_prefix and not mutate_suffix:
        mutate_prefix = True   # guarantee at least one change

    prefix = _pick(vocab["prefix"]) if mutate_prefix else parent.prefix
    suffix = _pick(vocab["suffix"]) if mutate_suffix else parent.suffix

    name_tag = strategy.title()
    return ResearchPersona(
        persona_id=_new_id(),
        name=f"{name_tag} Challenger",
        strategy=strategy,
        prefix=prefix,
        suffix=suffix,
        fitness=0.5,
        search_count=0,
        hit_count=0,
        active=True,
        generation=parent.generation + 1,
        parent_id=parent.persona_id,
        created_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# ResearchSwarm
# ---------------------------------------------------------------------------

class ResearchSwarm:
    """
    Manages a pool of active ResearchPersona objects, runs multi-perspective
    KB searches, and continuously evolves the pool via NEAT-style competition.

    Parameters
    ----------
    data_dir
        Root nexus_data directory.  Defaults to config value.
    max_active
        Maximum number of active personas at any time (default 5).
    competition_interval
        After this many individual persona-searches, trigger a competition
        round where the weakest active persona faces a new challenger
        (default 20).
    min_samples_to_compete
        A persona must have at least this many searches before it can be
        eliminated (protects new challengers from instant death).
    """

    _SWARM_DIR    = "swarm"
    _PERSONAS_FILE = "personas.jsonl"
    _ACTIVE_FILE   = "active.json"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_active: int = 5,
        competition_interval: int = 20,
        min_samples_to_compete: int = 5,
    ):
        base = Path(data_dir) if data_dir else Path(_config.nexus_data_path)
        self._dir = base / self._SWARM_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        self._personas_path = self._dir / self._PERSONAS_FILE
        self._active_path   = self._dir / self._ACTIVE_FILE

        self.max_active            = max_active
        self.competition_interval  = competition_interval
        self.min_samples_to_compete = min_samples_to_compete

        # Running counter of total individual-persona searches since last
        # evolution trigger.
        self._searches_since_evolve: int = 0

        self._ensure_seeded()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> List[ResearchPersona]:
        if not self._personas_path.exists():
            return []
        personas: List[ResearchPersona] = []
        with open(self._personas_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    personas.append(ResearchPersona.from_dict(json.loads(line)))
                except Exception:
                    pass
        return personas

    def _save_all(self, personas: List[ResearchPersona]) -> None:
        with open(self._personas_path, "w", encoding="utf-8") as f:
            for p in personas:
                f.write(json.dumps(p.to_dict()) + "\n")

    def _append(self, persona: ResearchPersona) -> None:
        with open(self._personas_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(persona.to_dict()) + "\n")

    def _load_active_ids(self) -> List[str]:
        if not self._active_path.exists():
            return []
        try:
            with open(self._active_path, encoding="utf-8") as f:
                return json.load(f).get("active_ids", [])
        except Exception:
            return []

    def _save_active_ids(self, ids: List[str]) -> None:
        with open(self._active_path, "w", encoding="utf-8") as f:
            json.dump({"active_ids": ids}, f)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _ensure_seeded(self) -> None:
        """Seed default personas on first run."""
        if self._personas_path.exists() and self._active_path.exists():
            active_ids = self._load_active_ids()
            if active_ids:
                return   # already seeded

        personas = []
        for name, strategy in _SEED_PERSONAS[: self.max_active]:
            p = _seed_persona(name, strategy, generation=0)
            personas.append(p)
            self._append(p)

        self._save_active_ids([p.persona_id for p in personas])
        print(
            f"[swarm] Seeded {len(personas)} research personas: "
            + ", ".join(p.name for p in personas)
        )

    # ------------------------------------------------------------------
    # Active persona access
    # ------------------------------------------------------------------

    def get_active(self) -> List[ResearchPersona]:
        """Return the currently active personas (ordered by persona_id)."""
        active_ids = set(self._load_active_ids())
        all_p = self._load_all()
        by_id = {p.persona_id: p for p in all_p}
        return [by_id[pid] for pid in active_ids if pid in by_id]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        kb_search_fn: Callable[[str, int], List[Dict[str, Any]]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run all active personas against the KB and return merged results.

        Each persona reformulates *query* and calls *kb_search_fn(reformulated,
        per_persona_k)*.  Results are deduplicated (by chunk_id or text hash),
        then re-ranked by score before the top *top_k* are returned.

        Persona fitness is updated in-place and the full list is persisted.

        Parameters
        ----------
        query
            The original, unexpanded user query.
        kb_search_fn
            Callable matching ``search_knowledge_base(query, data_dir, top_k)``
            signature — the swarm caller should provide a lambda that already
            binds data_dir.
        top_k
            Number of merged chunks to return.

        Returns
        -------
        List of chunk dicts (same schema as the raw KB search results).
        """
        active = self.get_active()
        if not active:
            # Fallback: search once with original query
            try:
                return kb_search_fn(query, top_k)
            except Exception:
                return []

        # Spread the per-persona budget; give each at least 3 results
        per_k = max(3, (top_k * 2) // len(active))

        # chunk_id/text-hash → (chunk_dict, contributing_persona_ids)
        seen:         Dict[str, Dict[str, Any]] = {}
        contributors: Dict[str, List[str]]      = defaultdict(list)

        for persona in active:
            reformulated = persona.reformulate(query)
            try:
                raw_results = kb_search_fn(reformulated, per_k)
            except Exception:
                raw_results = []

            for chunk in raw_results:
                key = chunk.get("chunk_id") or _text_hash(chunk.get("text", ""))
                if key not in seen:
                    seen[key] = chunk
                else:
                    # Keep the copy with the higher score
                    existing_score = seen[key].get("score", 0.0)
                    new_score      = chunk.get("score", 0.0)
                    if new_score > existing_score:
                        seen[key] = chunk
                contributors[key].append(persona.persona_id)

        # Determine hit/miss per persona
        all_keys        = set(seen.keys())
        persona_hit_ids = {
            pid
            for key, pids in contributors.items()
            for pid in pids
            if key in all_keys
        }
        # A persona "hit" if at least one of its chunks makes the merged pool
        hit_ids = set()
        if seen:
            # Take top_k by score to identify which chunks actually made it
            ranked = sorted(seen.values(), key=lambda c: c.get("score", 0.0), reverse=True)
            final  = ranked[:top_k]
            final_keys = {
                c.get("chunk_id") or _text_hash(c.get("text", ""))
                for c in final
            }
            for key in final_keys:
                for pid in contributors.get(key, []):
                    hit_ids.add(pid)
        else:
            final = []

        # Update fitness on all personas and persist
        all_personas = self._load_all()
        by_id = {p.persona_id: p for p in all_personas}
        for persona in active:
            if persona.persona_id in by_id:
                by_id[persona.persona_id].record_search(
                    hit=persona.persona_id in hit_ids
                )
                self._searches_since_evolve += 1

        self._save_all(list(by_id.values()))

        # Maybe trigger an evolution round
        if self._searches_since_evolve >= self.competition_interval:
            self._searches_since_evolve = 0
            self._run_competition()

        return final

    # ------------------------------------------------------------------
    # Evolution: continuous competitive elimination
    # ------------------------------------------------------------------

    def _run_competition(self) -> Optional[str]:
        """
        Replace the weakest active persona (if it has enough samples) with a
        challenger bred from the strongest active persona.

        Returns the new challenger's persona_id, or None if no elimination
        happened (all personas still have too few samples).
        """
        all_personas = self._load_all()
        by_id        = {p.persona_id: p for p in all_personas}
        active_ids   = self._load_active_ids()
        active       = [by_id[pid] for pid in active_ids if pid in by_id]

        if not active:
            return None

        # Need at least one persona with enough samples to eliminate
        eligible = [
            p for p in active
            if p.search_count >= self.min_samples_to_compete
        ]
        if not eligible:
            return None

        # Identify the weakest (lowest fitness among eligible)
        weakest = min(eligible, key=lambda p: p.fitness)

        # Breed a challenger from the strongest active persona
        strongest = max(active, key=lambda p: p.fitness)
        challenger = mutate_persona(strongest)

        # Eliminate the weakest; insert challenger
        by_id[weakest.persona_id].active = False
        by_id[challenger.persona_id] = challenger

        new_active_ids = [
            pid for pid in active_ids if pid != weakest.persona_id
        ]
        new_active_ids.append(challenger.persona_id)

        self._save_all(list(by_id.values()))
        self._save_active_ids(new_active_ids)

        print(
            f"[swarm] Evolution: eliminated '{weakest.name}' "
            f"(fitness={weakest.fitness:.3f} over {weakest.search_count} searches) "
            f"→ challenger '{challenger.name}' "
            f"(gen {challenger.generation}, parent='{strongest.name}')"
        )
        return challenger.persona_id

    def force_evolve(self) -> Dict[str, Any]:
        """
        Manually trigger a competition round regardless of the counter.
        Returns a summary dict.
        """
        prev_active = self.get_active()
        new_id = self._run_competition()
        curr_active = self.get_active()
        return {
            "eliminated": [
                p.to_dict() for p in prev_active
                if p.persona_id not in {a.persona_id for a in curr_active}
            ],
            "challenger_id": new_id,
            "active_now": [p.to_dict() for p in curr_active],
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary dict for the /swarm API endpoint."""
        all_personas  = self._load_all()
        active_ids    = set(self._load_active_ids())
        active        = [p for p in all_personas if p.persona_id in active_ids]
        eliminated    = [p for p in all_personas if p.persona_id not in active_ids]

        return {
            "active_count":     len(active),
            "eliminated_count": len(eliminated),
            "total_personas":   len(all_personas),
            "competition_interval": self.competition_interval,
            "searches_since_evolve": self._searches_since_evolve,
            "active_personas": [
                {
                    "persona_id":   p.persona_id,
                    "name":         p.name,
                    "strategy":     p.strategy,
                    "prefix":       p.prefix,
                    "suffix":       p.suffix,
                    "fitness":      p.fitness,
                    "search_count": p.search_count,
                    "hit_count":    p.hit_count,
                    "generation":   p.generation,
                }
                for p in sorted(active, key=lambda p: p.fitness, reverse=True)
            ],
            "top_eliminated": [
                {
                    "persona_id":   p.persona_id,
                    "name":         p.name,
                    "strategy":     p.strategy,
                    "fitness":      p.fitness,
                    "search_count": p.search_count,
                    "generation":   p.generation,
                }
                for p in sorted(eliminated, key=lambda p: p.fitness, reverse=True)[:5]
            ],
        }


# ---------------------------------------------------------------------------
# Small utility
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """Stable 16-char hash used when chunk_id is absent."""
    import hashlib
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]
