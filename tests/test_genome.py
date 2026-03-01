"""
Tests for nexus_core_genome.py

Covers: PromptGenome serialisation, GenomeStore persistence and fitness recording,
mutate(), crossover(), and evolve().
"""

import json
import random
import pytest
from pathlib import Path

from nexus_core_genome import (
    DEFAULT_GENES,
    GENE_SPEC,
    PromptGenome,
    GenomeStore,
    mutate,
    crossover,
    evolve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(generation: int = 0, fitness: float = 0.0, samples: int = 0,
                 genome_id: str = "aaaabbbb") -> PromptGenome:
    return PromptGenome(
        genome_id=genome_id,
        generation=generation,
        genes=DEFAULT_GENES.copy(),
        fitness=fitness,
        fitness_samples=samples,
        parent_ids=[],
        mutations=[],
        created_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# DEFAULT_GENES and GENE_SPEC sanity
# ---------------------------------------------------------------------------

def test_default_genes_keys_match_gene_spec():
    assert set(DEFAULT_GENES.keys()) == set(GENE_SPEC.keys())


def test_default_genes_values_within_bounds():
    for gene, val in DEFAULT_GENES.items():
        spec = GENE_SPEC[gene]
        if spec["type"] in ("float", "int"):
            assert spec["min"] <= val <= spec["max"], (
                f"{gene}={val} out of [{spec['min']}, {spec['max']}]"
            )
        elif spec["type"] == "choice":
            assert val in spec["options"]


# ---------------------------------------------------------------------------
# PromptGenome serialisation
# ---------------------------------------------------------------------------

def test_to_dict_contains_all_fields():
    g = _make_genome()
    d = g.to_dict()
    for key in ("genome_id", "generation", "genes", "fitness",
                "fitness_samples", "parent_ids", "mutations", "created_at"):
        assert key in d


def test_from_dict_round_trip():
    g = _make_genome(generation=2, fitness=0.75, samples=4)
    assert PromptGenome.from_dict(g.to_dict()) == g


def test_from_dict_missing_optional_fields():
    d = _make_genome().to_dict()
    del d["parent_ids"]
    del d["mutations"]
    g = PromptGenome.from_dict(d)
    assert g.parent_ids == []
    assert g.mutations == []


# ---------------------------------------------------------------------------
# GenomeStore — seeding and persistence
# ---------------------------------------------------------------------------

def test_store_seeds_generation_0_on_first_call(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    assert g.generation == 0
    assert g.fitness == 0.0
    assert g.genes == DEFAULT_GENES


def test_store_creates_neat_directory(tmp_path):
    GenomeStore(data_dir=str(tmp_path)).get_active_genome()
    assert (tmp_path / "neat").is_dir()


def test_store_persists_seed_to_jsonl(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    lines = (tmp_path / "neat" / "genomes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["genome_id"] == g.genome_id


def test_store_active_pointer_written(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    data = json.loads((tmp_path / "neat" / "active_genome.json").read_text())
    assert data["genome_id"] == g.genome_id


def test_store_returns_same_active_on_second_call(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g1 = store.get_active_genome()
    g2 = store.get_active_genome()
    assert g1.genome_id == g2.genome_id


def test_store_two_instances_share_data(tmp_path):
    store_a = GenomeStore(data_dir=str(tmp_path))
    gid = store_a.get_active_genome().genome_id
    store_b = GenomeStore(data_dir=str(tmp_path))
    assert store_b.get_active_genome().genome_id == gid


def test_store_get_all_returns_seeded_genome(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    store.get_active_genome()
    assert len(store.get_all()) == 1


# ---------------------------------------------------------------------------
# GenomeStore — record_fitness
# ---------------------------------------------------------------------------

def test_record_fitness_thumbs_up(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    updated = store.record_fitness(g.genome_id, score=1)
    assert updated is not None
    assert updated.fitness == 1.0
    assert updated.fitness_samples == 1


def test_record_fitness_thumbs_down(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    updated = store.record_fitness(g.genome_id, score=-1)
    assert updated.fitness == 0.0
    assert updated.fitness_samples == 1


def test_record_fitness_neutral(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    updated = store.record_fitness(g.genome_id, score=0)
    assert updated.fitness == 0.5
    assert updated.fitness_samples == 1


def test_record_fitness_running_average(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    store.record_fitness(g.genome_id, score=1)   # 1.0
    store.record_fitness(g.genome_id, score=-1)  # avg of 1.0 and 0.0 = 0.5
    updated = store.record_fitness(g.genome_id, score=1)   # avg of 1.0, 0.0, 1.0 = 0.6667
    assert updated.fitness_samples == 3
    assert abs(updated.fitness - round(2 / 3, 4)) < 0.001


def test_record_fitness_persists_update(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    store.record_fitness(g.genome_id, score=1)
    store2 = GenomeStore(data_dir=str(tmp_path))
    reloaded = store2.get_active_genome()
    assert reloaded.fitness == 1.0


def test_record_fitness_unknown_id_returns_none(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    store.get_active_genome()
    result = store.record_fitness("nonexistent", score=1)
    assert result is None


# ---------------------------------------------------------------------------
# GenomeStore — get_stats
# ---------------------------------------------------------------------------

def test_get_stats_shape(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    store.get_active_genome()
    stats = store.get_stats()
    for key in ("total_genomes", "generations", "current_generation",
                "active_genome_id", "active_fitness", "active_fitness_samples",
                "active_genes"):
        assert key in stats


def test_get_stats_after_seed(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    stats = store.get_stats()
    assert stats["total_genomes"] == 1
    assert stats["current_generation"] == 0
    assert stats["active_genome_id"] == g.genome_id
    assert stats["active_genes"] == DEFAULT_GENES


# ---------------------------------------------------------------------------
# mutate()
# ---------------------------------------------------------------------------

def test_mutate_returns_new_object():
    g = _make_genome()
    m = mutate(g, mutation_rate=1.0)
    assert m is not g


def test_mutate_new_genome_id():
    g = _make_genome()
    m = mutate(g, mutation_rate=1.0)
    assert m.genome_id != g.genome_id


def test_mutate_increments_generation():
    g = _make_genome(generation=3)
    m = mutate(g)
    assert m.generation == 4


def test_mutate_parent_ids_contains_original():
    g = _make_genome()
    m = mutate(g)
    assert g.genome_id in m.parent_ids


def test_mutate_zeroes_fitness():
    g = _make_genome(fitness=0.9, samples=10)
    m = mutate(g)
    assert m.fitness == 0.0
    assert m.fitness_samples == 0


def test_mutate_at_least_one_change():
    random.seed(42)
    g = _make_genome()
    m = mutate(g, mutation_rate=1.0)
    assert len(m.mutations) >= 1


def test_mutate_genes_within_bounds():
    random.seed(0)
    g = _make_genome()
    for _ in range(20):
        g = mutate(g, mutation_rate=1.0)
    for gene, val in g.genes.items():
        spec = GENE_SPEC[gene]
        if spec["type"] == "float":
            assert spec["min"] <= val <= spec["max"]
        elif spec["type"] == "int":
            assert spec["min"] <= val <= spec["max"]
        elif spec["type"] == "choice":
            assert val in spec["options"]


def test_mutate_low_rate_still_changes_at_least_one_gene():
    random.seed(99)
    g = _make_genome()
    m = mutate(g, mutation_rate=0.001)
    assert m.genes != g.genes or len(m.mutations) >= 1


# ---------------------------------------------------------------------------
# crossover()
# ---------------------------------------------------------------------------

def test_crossover_returns_new_object():
    a = _make_genome(genome_id="aaaa1111")
    b = _make_genome(genome_id="bbbb2222")
    c = crossover(a, b)
    assert c is not a and c is not b


def test_crossover_has_both_parent_ids():
    a = _make_genome(genome_id="aaaa1111")
    b = _make_genome(genome_id="bbbb2222")
    c = crossover(a, b)
    assert "aaaa1111" in c.parent_ids
    assert "bbbb2222" in c.parent_ids


def test_crossover_genes_from_parents():
    a = _make_genome(genome_id="aaaa1111")
    b = _make_genome(genome_id="bbbb2222")
    # Give b distinct gene values so we can tell them apart
    distinct = {}
    for k, spec in GENE_SPEC.items():
        if spec["type"] == "float":
            distinct[k] = spec["max"]
        elif spec["type"] == "int":
            distinct[k] = spec["max"]
        elif spec["type"] == "bool":
            distinct[k] = not DEFAULT_GENES[k]
        else:  # choice
            distinct[k] = spec["options"][-1]
    b.genes = distinct
    c = crossover(a, b)
    for gene in GENE_SPEC:
        assert c.genes[gene] in (a.genes[gene], b.genes[gene])


def test_crossover_zeroes_fitness():
    a = _make_genome(fitness=0.8, samples=5, genome_id="aaaa1111")
    b = _make_genome(fitness=0.9, samples=7, genome_id="bbbb2222")
    c = crossover(a, b)
    assert c.fitness == 0.0
    assert c.fitness_samples == 0


def test_crossover_generation_is_max_plus_one():
    a = _make_genome(generation=3, genome_id="aaaa1111")
    b = _make_genome(generation=5, genome_id="bbbb2222")
    c = crossover(a, b)
    assert c.generation == 6


# ---------------------------------------------------------------------------
# evolve()
# ---------------------------------------------------------------------------

def test_evolve_no_rated_genomes_returns_current(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    pop, active = evolve(store, min_fitness_samples=1)
    assert len(pop) == 1
    assert active.genome_id == g.genome_id


def test_evolve_produces_new_population(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    store.record_fitness(g.genome_id, 1)
    store.record_fitness(g.genome_id, 1)
    store.record_fitness(g.genome_id, 1)
    pop, active = evolve(store, population_size=5, min_fitness_samples=3)
    assert len(pop) == 5


def test_evolve_new_active_is_set(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    for _ in range(3):
        store.record_fitness(g.genome_id, 1)
    _, new_active = evolve(store, min_fitness_samples=3)
    assert store.get_active_genome().genome_id == new_active.genome_id


def test_evolve_new_generation_is_higher(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    for _ in range(3):
        store.record_fitness(g.genome_id, 1)
    _, new_active = evolve(store, min_fitness_samples=3)
    assert new_active.generation > g.generation


def test_evolve_offspring_added_to_store(tmp_path):
    store = GenomeStore(data_dir=str(tmp_path))
    g = store.get_active_genome()
    for _ in range(3):
        store.record_fitness(g.genome_id, 1)
    pop, _ = evolve(store, population_size=4, min_fitness_samples=3)
    assert len(store.get_all()) == 1 + 4  # seed + new population
