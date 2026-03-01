"""
The Nexus Core - NEAT Prompt Genome

Defines the evolvable genome schema and the persistence / evolution layer.
Zero external dependencies — stdlib only.

Genomes are stored in  nexus_data/neat/genomes.jsonl  (append-on-write).
The active genome pointer lives in  nexus_data/neat/active_genome.json.

Typical usage
-------------
    from nexus_core_genome import GenomeStore, mutate, crossover, evolve

    store  = GenomeStore()
    genome = store.get_active_genome()   # seeds generation 0 on first call

    # After a /feedback rating comes in:
    store.record_fitness(genome.genome_id, rating=1)

    # When enough samples have been collected:
    new_pop, new_active = evolve(store)
"""

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus_core_config import config as _config


# ---------------------------------------------------------------------------
# Gene definitions
# ---------------------------------------------------------------------------

#: Default gene values — used when seeding generation 0
DEFAULT_GENES: Dict[str, Any] = {
    "temperature":            0.7,
    "search_top_k":           5,
    "use_recursive":          False,
    "system_prompt_style":    "balanced",  # concise | detailed | analytical | balanced
    "chunk_diversity_weight": 0.5,
    "context_window_size":    4096,
}

#: Per-gene mutation spec
GENE_SPEC: Dict[str, Dict[str, Any]] = {
    "temperature": {
        "type": "float", "min": 0.0, "max": 1.5,
    },
    "search_top_k": {
        "type": "int", "min": 1, "max": 15,
    },
    "use_recursive": {
        "type": "bool",
    },
    "system_prompt_style": {
        "type": "choice",
        "options": ["concise", "detailed", "analytical", "balanced"],
    },
    "chunk_diversity_weight": {
        "type": "float", "min": 0.0, "max": 1.0,
    },
    "context_window_size": {
        "type": "int", "min": 512, "max": 8192,
    },
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PromptGenome:
    """
    An evolvable set of prompt-generation parameters.

    Attributes
    ----------
    genome_id       : 8-char hex identifier
    generation      : NEAT generation number (0 = seed)
    genes           : dict of evolvable parameter values
    fitness         : running average of normalised feedback scores (0.0–1.0)
    fitness_samples : number of feedback events that contributed to fitness
    parent_ids      : [] for seed genomes, [parent_id] for mutations,
                      [a_id, b_id] for crossover offspring
    mutations       : human-readable description of changes from parent(s)
    created_at      : ISO 8601 UTC timestamp
    """

    genome_id: str
    generation: int
    genes: Dict[str, Any]
    fitness: float
    fitness_samples: int
    parent_ids: List[str]
    mutations: List[str]
    created_at: str

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id":        self.genome_id,
            "generation":       self.generation,
            "genes":            self.genes,
            "fitness":          self.fitness,
            "fitness_samples":  self.fitness_samples,
            "parent_ids":       self.parent_ids,
            "mutations":        self.mutations,
            "created_at":       self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PromptGenome":
        return PromptGenome(
            genome_id=d["genome_id"],
            generation=d["generation"],
            genes=d["genes"],
            fitness=d["fitness"],
            fitness_samples=d["fitness_samples"],
            parent_ids=d.get("parent_ids", []),
            mutations=d.get("mutations", []),
            created_at=d["created_at"],
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class GenomeStore:
    """
    Append-on-write JSONL store for PromptGenome objects.
    Thread-safe for single-process use.
    """

    _NEAT_DIR    = "neat"
    _GENOMES_FILE = "genomes.jsonl"
    _ACTIVE_FILE  = "active_genome.json"

    def __init__(self, data_dir: Optional[str] = None):
        base = Path(data_dir) if data_dir else Path(_config.nexus_data_path)
        self._neat_dir = base / self._NEAT_DIR
        self._neat_dir.mkdir(parents=True, exist_ok=True)
        self._genomes_path = self._neat_dir / self._GENOMES_FILE
        self._active_path  = self._neat_dir / self._ACTIVE_FILE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> List[PromptGenome]:
        if not self._genomes_path.exists():
            return []
        genomes: List[PromptGenome] = []
        with open(self._genomes_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    genomes.append(PromptGenome.from_dict(json.loads(line)))
                except Exception:
                    pass
        return genomes

    def _append(self, genome: PromptGenome) -> None:
        with open(self._genomes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(genome.to_dict()) + "\n")

    def _save_active(self, genome_id: str) -> None:
        with open(self._active_path, "w", encoding="utf-8") as f:
            json.dump({"genome_id": genome_id}, f)

    def _load_active_id(self) -> Optional[str]:
        if not self._active_path.exists():
            return None
        try:
            with open(self._active_path, encoding="utf-8") as f:
                return json.load(f).get("genome_id")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_genome(self) -> PromptGenome:
        """
        Return the active genome.  Seeds generation 0 on first call.
        """
        active_id = self._load_active_id()
        if active_id:
            lookup = {g.genome_id: g for g in self._load_all()}
            if active_id in lookup:
                return lookup[active_id]
        return self._seed()

    def _seed(self) -> PromptGenome:
        genome = PromptGenome(
            genome_id=str(uuid.uuid4()).replace("-", "")[:8],
            generation=0,
            genes=DEFAULT_GENES.copy(),
            fitness=0.0,
            fitness_samples=0,
            parent_ids=[],
            mutations=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append(genome)
        self._save_active(genome.genome_id)
        print(f"[genome] Seeded generation 0 — id: {genome.genome_id}")
        return genome

    def record_fitness(self, genome_id: str, score: float) -> Optional[PromptGenome]:
        """
        Update the genome's cumulative fitness with one new sample.

        score — raw feedback value (+1 thumbs up, -1 thumbs down, 0 neutral).
        Internally normalised to [0, 1]:  +1 → 1.0,  0 → 0.5,  -1 → 0.0.

        Rewrites the genomes file with the updated record.
        Returns the updated genome, or None if genome_id is not found.
        """
        normalised = (score + 1.0) / 2.0   # maps [-1,1] → [0,1]
        genomes = self._load_all()

        target: Optional[PromptGenome] = None
        for g in reversed(genomes):
            if g.genome_id == genome_id:
                target = g
                break

        if target is None:
            return None

        n = target.fitness_samples
        target.fitness = round((target.fitness * n + normalised) / (n + 1), 4)
        target.fitness_samples = n + 1

        # Rewrite with updated record
        by_id = {g.genome_id: g for g in genomes}
        by_id[target.genome_id] = target
        with open(self._genomes_path, "w", encoding="utf-8") as f:
            for g in by_id.values():
                f.write(json.dumps(g.to_dict()) + "\n")

        return target

    def get_all(self) -> List[PromptGenome]:
        return self._load_all()

    def get_stats(self) -> Dict[str, Any]:
        genomes   = self._load_all()
        active_id = self._load_active_id()
        active    = next((g for g in genomes if g.genome_id == active_id), None)
        gens      = sorted({g.generation for g in genomes})
        return {
            "total_genomes":         len(genomes),
            "generations":           gens,
            "current_generation":    max(gens) if gens else 0,
            "active_genome_id":      active_id,
            "active_fitness":        round(active.fitness, 4) if active else 0.0,
            "active_fitness_samples": active.fitness_samples if active else 0,
            "active_genes":          active.genes if active else {},
        }


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:8]


def _force_mutate_one(genes: Dict[str, Any], mutation_log: List[str]) -> None:
    """Mutate exactly one randomly chosen gene in-place."""
    gene = random.choice(list(GENE_SPEC.keys()))
    spec = GENE_SPEC[gene]
    old_val = genes.get(gene, DEFAULT_GENES[gene])

    if spec["type"] == "bool":
        new_val = not old_val
    elif spec["type"] == "choice":
        opts = [o for o in spec["options"] if o != old_val]
        new_val = random.choice(opts) if opts else old_val
    elif spec["type"] == "float":
        rng   = spec["max"] - spec["min"]
        new_val = round(
            max(spec["min"], min(spec["max"], old_val + random.gauss(0, rng * 0.15))), 3
        )
    else:  # int
        step    = max(1, (spec["max"] - spec["min"]) // 10)
        new_val = max(spec["min"], min(spec["max"], old_val + random.choice([-step, step])))

    genes[gene] = new_val
    mutation_log.append(f"{gene}: {old_val} -> {new_val}")


def mutate(genome: PromptGenome, mutation_rate: float = 0.2) -> PromptGenome:
    """
    Return a new PromptGenome derived from *genome* by random per-gene
    perturbation.  At least one gene is guaranteed to change.

    mutation_rate — independent probability each gene is mutated.
    """
    new_genes = genome.genes.copy()
    mutation_log: List[str] = []

    for gene, spec in GENE_SPEC.items():
        if random.random() > mutation_rate:
            continue

        old_val = new_genes.get(gene, DEFAULT_GENES[gene])

        if spec["type"] == "float":
            rng     = spec["max"] - spec["min"]
            new_val = round(
                max(spec["min"], min(spec["max"], old_val + random.gauss(0, rng * 0.15))), 3
            )
        elif spec["type"] == "int":
            step    = max(1, (spec["max"] - spec["min"]) // 10)
            new_val = max(spec["min"], min(spec["max"], old_val + random.randint(-step, step)))
        elif spec["type"] == "bool":
            new_val = not old_val
        elif spec["type"] == "choice":
            opts    = [o for o in spec["options"] if o != old_val]
            new_val = random.choice(opts) if opts else old_val
        else:
            continue

        if new_val != old_val:
            mutation_log.append(f"{gene}: {old_val} -> {new_val}")
            new_genes[gene] = new_val

    # Guarantee at least one change
    if not mutation_log:
        _force_mutate_one(new_genes, mutation_log)

    return PromptGenome(
        genome_id=_new_id(),
        generation=genome.generation + 1,
        genes=new_genes,
        fitness=0.0,
        fitness_samples=0,
        parent_ids=[genome.genome_id],
        mutations=mutation_log,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover(parent_a: PromptGenome, parent_b: PromptGenome) -> PromptGenome:
    """
    Produce an offspring by randomly drawing each gene from one of two parents.
    """
    new_genes: Dict[str, Any] = {}
    cross_log: List[str] = []

    for gene in GENE_SPEC:
        a_val = parent_a.genes.get(gene, DEFAULT_GENES[gene])
        b_val = parent_b.genes.get(gene, DEFAULT_GENES[gene])
        chosen = random.choice([a_val, b_val])
        new_genes[gene] = chosen
        if chosen == b_val and a_val != b_val:
            cross_log.append(f"{gene} from {parent_b.genome_id}")

    generation = max(parent_a.generation, parent_b.generation) + 1

    return PromptGenome(
        genome_id=_new_id(),
        generation=generation,
        genes=new_genes,
        fitness=0.0,
        fitness_samples=0,
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
        mutations=cross_log if cross_log else ["uniform inherit"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

def evolve(
    store: GenomeStore,
    population_size: int = 8,
    elitism_count: int = 3,
    mutation_rate: float = 0.2,
    min_fitness_samples: int = 3,
) -> Tuple[List[PromptGenome], PromptGenome]:
    """
    Run one NEAT generation cycle.

    1. Load all genomes that have >= min_fitness_samples feedback events.
    2. Rank by fitness (descending).
    3. Carry the top *elitism_count* forward unchanged (elitism).
    4. Fill the remainder of *population_size* via crossover + mutation.
    5. Persist all new genomes; set the elitism leader as active.

    Returns
    -------
    (new_population, new_active_genome)

    If no evaluated genomes exist yet, returns ([current_active], current_active)
    without writing anything new.
    """
    all_genomes = store.get_all()
    evaluated   = [g for g in all_genomes if g.fitness_samples >= min_fitness_samples]

    if not evaluated:
        active = store.get_active_genome()
        return [active], active

    evaluated.sort(key=lambda g: g.fitness, reverse=True)

    elite      = evaluated[:elitism_count]
    new_pop: List[PromptGenome] = []

    # Carry elite forward (re-issued with fresh IDs and zeroed fitness)
    for g in elite:
        heir = PromptGenome(
            genome_id=_new_id(),
            generation=g.generation + 1,
            genes=g.genes.copy(),
            fitness=0.0,
            fitness_samples=0,
            parent_ids=[g.genome_id],
            mutations=["elitism carry-forward"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        new_pop.append(heir)

    # Fill remainder with crossover offspring that are also mutated
    pool = elite[:min(5, len(elite))]
    while len(new_pop) < population_size:
        if len(pool) >= 2:
            pa, pb = random.sample(pool, 2)
            child  = crossover(pa, pb)
        else:
            child = PromptGenome(
                genome_id=_new_id(),
                generation=pool[0].generation + 1,
                genes=pool[0].genes.copy(),
                fitness=0.0,
                fitness_samples=0,
                parent_ids=[pool[0].genome_id],
                mutations=[],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        new_pop.append(mutate(child, mutation_rate=mutation_rate))

    for g in new_pop:
        store._append(g)

    new_active = new_pop[0]   # elitism leader
    store._save_active(new_active.genome_id)

    print(
        f"[genome] Evolved to generation {new_active.generation} — "
        f"{len(new_pop)} genomes, active: {new_active.genome_id}"
    )
    return new_pop, new_active
