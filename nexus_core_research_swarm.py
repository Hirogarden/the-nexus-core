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
import threading
import time
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

# EMA decay factor for fitness tracking.
# Each new search sample gets this weight; prior history gets (1 - _EMA_ALPHA).
# 0.15 means ~15 % of weight on the newest result, letting fitness adapt to
# environmental changes in roughly 10-20 searches rather than hundreds.
_EMA_ALPHA: float = 0.15

# Default seed personas — one per strategy except comparative (seeded later)
_SEED_PERSONAS = [
    ("Technical Analyst",    "technical"),
    ("Broad Context Seeker", "broad"),
    ("Critical Examiner",    "skeptical"),
    ("Practical Applier",    "practical"),
    ("Historical Researcher","historical"),
]


# ---------------------------------------------------------------------------
# Warm-up state
# ---------------------------------------------------------------------------

@dataclass
class WarmupState:
    """
    Tracks the progress of a background warm-up session.

    Attributes
    ----------
    running               — True while the warmup thread is active.
    iterations_completed  — Number of search iterations finished so far.
    iterations_target     — The max_iterations ceiling requested.
    seconds_target        — The max_seconds ceiling requested.
    evolutions_triggered  — Competition rounds that fired during this session.
    seed_query_count      — Number of distinct seed queries available.
    started_at            — ISO timestamp of session start, or None.
    finished_at           — ISO timestamp of session end, or None.
    stop_reason           — Why the session ended:
                              "iterations"      — hit max_iterations
                              "time"            — hit max_seconds
                              "stopped_by_user" — stop_event was set
                              "no_queries"      — empty query list
                              ""                — not yet finished
    """
    running:              bool           = False
    iterations_completed: int            = 0
    iterations_target:    int            = 0
    seconds_target:       float          = 0.0
    evolutions_triggered: int            = 0
    seed_query_count:     int            = 0
    started_at:           Optional[str]  = None
    finished_at:          Optional[str]  = None
    stop_reason:          str            = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running":              self.running,
            "iterations_completed": self.iterations_completed,
            "iterations_target":    self.iterations_target,
            "seconds_target":       self.seconds_target,
            "evolutions_triggered": self.evolutions_triggered,
            "seed_query_count":     self.seed_query_count,
            "started_at":           self.started_at,
            "finished_at":          self.finished_at,
            "stop_reason":          self.stop_reason,
        }


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
        """Update fitness after one search using an exponential moving average.

        EMA adapts ~10–20× faster than a running average when the query
        distribution or KB content changes, because recent samples always
        carry a fixed weight (_EMA_ALPHA) rather than diminishing 1/n weight.
        """
        self.search_count += 1
        if hit:
            self.hit_count += 1
        new_sample = 1.0 if hit else 0.0
        self.fitness = round(
            _EMA_ALPHA * new_sample + (1.0 - _EMA_ALPHA) * self.fitness, 4
        )

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
        min_samples_to_compete: int = 20,
    ):
        base = Path(data_dir) if data_dir else Path(_config.nexus_data_path)
        self._dir = base / self._SWARM_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        self._personas_path = self._dir / self._PERSONAS_FILE
        self._active_path   = self._dir / self._ACTIVE_FILE

        self.max_active            = max_active
        self.competition_interval  = competition_interval
        self.min_samples_to_compete = min_samples_to_compete

        # Re-entrant lock protecting all file read-modify-write sequences.
        # RLock (not Lock) is used so that search() can call _run_competition()
        # while already holding the lock without deadlocking.
        self._lock: threading.RLock = threading.RLock()

        # Running counter of total individual-persona searches since last
        # evolution trigger.
        self._searches_since_evolve: int = 0

        # Total number of competition rounds that have ever fired on this instance.
        # Used by warmup to count evolutions triggered during a session.
        self._evolution_count: int = 0

        # Warmup session state (in-memory only, not persisted)
        self._warmup_state: WarmupState = WarmupState()

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

        Thread safety
        -------------
        KB searches are performed outside the internal lock (they can be slow
        network or disk calls and should not block concurrent threads).  The
        subsequent fitness update and file write are performed under
        ``self._lock`` (a re-entrant lock) so that concurrent calls from the
        warmup thread and the API request thread cannot corrupt the JSONL file.

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

        # --- KB searches: performed OUTSIDE the lock (slow I/O) ----------
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

        # Determine which chunks actually make the final top_k
        if seen:
            ranked    = sorted(seen.values(), key=lambda c: c.get("score", 0.0), reverse=True)
            final     = ranked[:top_k]
            final_keys = {
                c.get("chunk_id") or _text_hash(c.get("text", ""))
                for c in final
            }
            hit_ids = {
                pid
                for key in final_keys
                for pid in contributors.get(key, [])
            }
        else:
            final   = []
            hit_ids = set()

        # --- Fitness update + persistence: under the lock ----------------
        with self._lock:
            all_personas = self._load_all()
            by_id = {p.persona_id: p for p in all_personas}
            for persona in active:
                if persona.persona_id in by_id:
                    by_id[persona.persona_id].record_search(
                        hit=persona.persona_id in hit_ids
                    )
                    self._searches_since_evolve += 1

            self._save_all(list(by_id.values()))

            # Maybe trigger an evolution round (RLock allows re-entry)
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
        challenger bred from the active pool.

        Improvements over naïve "eliminate weakest, clone strongest":

        Diversity-aware elimination
            Personas whose strategy is over-represented face extra elimination
            pressure (up to 0.15 fitness penalty per density unit), so a
            dominant strategy can't simply crowd out all others through luck.

        Probabilistic parent selection
            80 % of the time the challenger is bred from the strongest persona
            (exploitation); 20 % of the time from a random active persona
            (exploration), preventing the gene pool from converging to a single
            lineage within a few dozen generations.

        Periodic strategy revival
            Every 5 competition rounds, if any of the 7 strategy types is
            absent from the active pool, a fresh seed persona for that strategy
            is injected as the challenger.  This ensures lost strategies can
            always re-enter competition.

        Returns the new challenger's persona_id, or None if no elimination
        happened (all personas still have too few samples).

        Thread safety: wraps the entire read-modify-write section in
        ``self._lock``.  When called from ``search()`` (which already holds
        the RLock), re-entry is safe.
        """
        with self._lock:
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

            # --- Diversity-aware weakest selection -----------------------
            # Penalise personas whose strategy dominates the active pool so
            # that a flood of one strategy type doesn't become unassailable.
            strategy_counts: Dict[str, int] = defaultdict(int)
            for p in active:
                strategy_counts[p.strategy] += 1
            n_active = len(active)

            def _elimination_score(p: "ResearchPersona") -> float:
                density = strategy_counts[p.strategy] / n_active
                # Penalty: up to 0.15 for a fully dominant strategy (all same)
                return p.fitness - 0.15 * density

            weakest = min(eligible, key=_elimination_score)

            # --- Probabilistic parent selection ---------------------------
            # 80 % exploitation (breed from strongest) to capitalise on good
            # genes; 20 % exploration (breed from random) to maintain lineage
            # diversity and prevent convergence to one ancestral line.
            if random.random() < 0.20:
                parent = random.choice(active)
            else:
                parent = max(active, key=lambda p: p.fitness)

            challenger = mutate_persona(parent)

            # --- Periodic strategy revival --------------------------------
            # Every 5 rounds, re-introduce a strategy type that has been
            # completely eliminated from the active pool.
            remaining_strategies = {
                p.strategy for p in active
                if p.persona_id != weakest.persona_id
            }
            remaining_strategies.add(challenger.strategy)
            missing_strategies = [s for s in _STRATEGIES if s not in remaining_strategies]

            if self._evolution_count % 5 == 0 and missing_strategies:
                revival_strategy = random.choice(missing_strategies)
                challenger = _seed_persona(
                    f"{revival_strategy.title()} Revival",
                    revival_strategy,
                    generation=challenger.generation,
                    parent_id=None,   # genuinely fresh genes, not a descendant
                )
                print(f"[swarm] Diversity revival: reintroducing '{revival_strategy}' strategy")

            # --- Apply elimination + insertion ----------------------------
            by_id[weakest.persona_id].active = False
            by_id[challenger.persona_id] = challenger

            new_active_ids = [
                pid for pid in active_ids if pid != weakest.persona_id
            ]
            new_active_ids.append(challenger.persona_id)

            self._save_all(list(by_id.values()))
            self._save_active_ids(new_active_ids)

            self._evolution_count += 1
            print(
                f"[swarm] Evolution #{self._evolution_count}: "
                f"eliminated '{weakest.name}' "
                f"(fitness={weakest.fitness:.3f}, {weakest.search_count} searches) "
                f"→ '{challenger.name}' "
                f"(gen {challenger.generation}, parent='{parent.name}')"
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
    # Warm-up
    # ------------------------------------------------------------------

    def run_warmup(
        self,
        queries: List[str],
        kb_search_fn: Callable[[str, int], List[Dict[str, Any]]],
        max_iterations: int = 50,
        max_seconds: float = 300.0,
        top_k: int = 5,
        stop_event: Optional[threading.Event] = None,
    ) -> "WarmupState":
        """
        Run warm-up iterations in the calling thread.

        Intended to be called from a background thread via
        ``BrainLikeAI.start_swarm_warmup()``.  Each iteration picks a random
        query from *queries*, calls ``search()``, and lets the normal fitness /
        competition machinery run.

        The loop terminates when any of these conditions is met (whichever
        comes first):

        * *max_iterations* iterations have completed  → stop_reason "iterations"
        * *max_seconds* wall-clock seconds have elapsed → stop_reason "time"
        * *stop_event* is set by the caller             → stop_reason "stopped_by_user"
        * *queries* is empty                            → stop_reason "no_queries"

        Thread safety
        -------------
        ``search()`` now acquires an internal RLock for its read-modify-write
        disk section, so this method is safe to run concurrently with normal
        API request handling.
        """
        state = self._warmup_state
        state.running              = True
        state.iterations_completed = 0
        state.iterations_target    = max_iterations
        state.seconds_target       = max_seconds
        state.seed_query_count     = len(queries)
        state.evolutions_triggered = 0
        state.started_at           = _now_iso()
        state.finished_at          = None
        state.stop_reason          = ""

        evolutions_before = self._evolution_count
        start_wall        = time.monotonic()

        try:
            if not queries:
                state.stop_reason = "no_queries"
                return state

            for i in range(max_iterations):
                # --- stop checks (before doing work) ---
                if stop_event is not None and stop_event.is_set():
                    state.stop_reason = "stopped_by_user"
                    break

                elapsed = time.monotonic() - start_wall
                if elapsed >= max_seconds:
                    state.stop_reason = "time"
                    break

                # --- one iteration ---
                query = random.choice(queries)
                try:
                    self.search(query, kb_search_fn, top_k=top_k)
                except Exception:
                    pass   # individual failures don't abort the session

                state.iterations_completed = i + 1
                state.evolutions_triggered = self._evolution_count - evolutions_before
            else:
                # Loop ran to completion without break
                state.stop_reason = "iterations"

        finally:
            state.running              = False
            state.finished_at          = _now_iso()
            state.evolutions_triggered = self._evolution_count - evolutions_before

        return state

    def get_warmup_status(self) -> Dict[str, Any]:
        """Return the current warmup session state as a plain dict."""
        return self._warmup_state.to_dict()

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
