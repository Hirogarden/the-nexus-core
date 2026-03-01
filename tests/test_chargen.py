"""
Tests for chargen_system.py — CharGenSystem and Persona.

Run with:
    pytest "c:/Users/hirog/The Nexus Core/tests/test_chargen.py" -v
"""

import json
import sys
import os
from pathlib import Path

# Ensure project root is importable regardless of invocation directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from chargen_system import CharGenSystem, Persona, PersonaTrait


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_system(tmp_path: Path) -> CharGenSystem:
    """Return a fresh CharGenSystem backed by a temporary directory."""
    return CharGenSystem(storage_path=str(tmp_path / "personas"))


def make_expert(system: CharGenSystem, name: str = "Dr. Smith") -> Persona:
    """Create a quick expert persona."""
    return system.generate_persona(name=name, template="expert")


# ===========================================================================
# 1 – Initialisation
# ===========================================================================

def test_init_creates_storage_directory(tmp_path):
    storage = tmp_path / "personas"
    assert not storage.exists()
    CharGenSystem(storage_path=str(storage))
    assert storage.is_dir()


def test_init_registers_five_templates(tmp_path):
    system = make_system(tmp_path)
    expected = {"expert", "companion", "analyst", "creative", "teacher"}
    assert set(system.persona_templates.keys()) == expected


# ===========================================================================
# 2 – generate_persona
# ===========================================================================

def test_generate_persona_returns_persona_instance(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)
    assert isinstance(persona, Persona)


def test_generate_persona_id_is_16_char_hex(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)
    assert len(persona.persona_id) == 16
    # Must be a valid lowercase hex string
    int(persona.persona_id, 16)  # raises ValueError if not valid hex


def test_generate_persona_saves_to_disk(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)
    expected_file = tmp_path / "personas" / f"{persona.persona_id}.json"
    assert expected_file.is_file()
    data = json.loads(expected_file.read_text())
    assert data["persona_id"] == persona.persona_id
    assert data["name"] == persona.name


def test_generate_persona_unknown_template_falls_back_to_expert(tmp_path):
    system = make_system(tmp_path)
    persona = system.generate_persona(name="Fallback", template="nonexistent_template")
    # expert role is "domain expert"
    assert persona.role == system.persona_templates["expert"]["role"]


def test_generate_persona_knowledge_domains_propagated(tmp_path):
    system = make_system(tmp_path)
    domains = ["machine learning", "statistics"]
    persona = system.generate_persona(
        name="Data Guru", template="analyst", knowledge_domains=domains
    )
    assert persona.knowledge_domains == domains


def test_generate_persona_backstory_propagated(tmp_path):
    system = make_system(tmp_path)
    backstory = "Born in a library, raised by algorithms."
    persona = system.generate_persona(name="Lore", backstory=backstory)
    assert persona.backstory == backstory


def test_generate_persona_custom_traits_merged(tmp_path):
    system = make_system(tmp_path)
    custom = {"humorous": PersonaTrait("humorous", 0.8, "Likes to joke")}
    persona = system.generate_persona(
        name="Wit", template="expert", custom_traits=custom
    )
    assert "humorous" in persona.traits
    assert pytest.approx(persona.traits["humorous"].value) == 0.8
    # Template traits should also be present
    assert "analytical" in persona.traits


# ===========================================================================
# 3 – get_persona
# ===========================================================================

def test_get_persona_returns_correct_persona(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system, name="Atlas")
    retrieved = system.get_persona(persona.persona_id)
    assert retrieved is persona


def test_get_persona_returns_none_for_missing_id(tmp_path):
    system = make_system(tmp_path)
    assert system.get_persona("deadbeef00000000") is None


# ===========================================================================
# 4 – list_personas
# ===========================================================================

def test_list_personas_active_only_excludes_deactivated(tmp_path):
    system = make_system(tmp_path)
    active = make_expert(system, name="Active")
    dormant = make_expert(system, name="Dormant")
    system.deactivate_persona(dormant.persona_id)

    results = system.list_personas(active_only=True)
    ids = [p.persona_id for p in results]
    assert active.persona_id in ids
    assert dormant.persona_id not in ids


def test_list_personas_active_only_false_includes_inactive(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system, name="Retired")
    system.deactivate_persona(persona.persona_id)

    results = system.list_personas(active_only=False)
    ids = [p.persona_id for p in results]
    assert persona.persona_id in ids


def test_list_personas_role_filter(tmp_path):
    system = make_system(tmp_path)
    system.generate_persona(name="Teach", template="teacher")
    system.generate_persona(name="Think", template="analyst")

    teachers = system.list_personas(active_only=True, role_filter="educator")
    assert all("educator" in p.role.lower() for p in teachers)
    assert len(teachers) >= 1


# ===========================================================================
# 5 – update_persona_traits
# ===========================================================================

def test_update_persona_traits_returns_true_on_success(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)
    result = system.update_persona_traits(persona.persona_id, {"analytical": 0.0})
    assert result is True


def test_update_persona_traits_clamping(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)

    # analytical starts at 0.9; adding +100 should clamp to 1.0
    system.update_persona_traits(persona.persona_id, {"analytical": 100.0})
    assert persona.traits["analytical"].value == pytest.approx(1.0)

    # formal starts at 0.7; subtracting 100 should clamp to 0.0
    system.update_persona_traits(persona.persona_id, {"formal": -100.0})
    assert persona.traits["formal"].value == pytest.approx(0.0)


def test_update_persona_traits_unknown_id_returns_false(tmp_path):
    system = make_system(tmp_path)
    result = system.update_persona_traits("0000000000000000", {"analytical": 0.1})
    assert result is False


# ===========================================================================
# 6 – record_interaction
# ===========================================================================

def test_record_interaction_increments_count(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)
    assert persona.interaction_count == 0

    system.record_interaction(persona.persona_id)
    assert persona.interaction_count == 1

    system.record_interaction(persona.persona_id)
    assert persona.interaction_count == 2


# ===========================================================================
# 7 – adapt_persona_to_context
# ===========================================================================

def test_adapt_persona_high_urgency(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)  # expert has "formal" trait at 0.7

    behavior = system.adapt_persona_to_context(
        persona.persona_id, {"urgency": "high"}
    )

    # formality should increase by 0.2 (from 0.7 -> 0.9)
    assert behavior["formality"] == pytest.approx(0.9)
    # verbosity should decrease by 0.3 (from 0.5 -> 0.2)
    assert behavior["verbosity"] == pytest.approx(0.2)


def test_adapt_persona_sad_mood_and_high_complexity(tmp_path):
    system = make_system(tmp_path)
    # Use expert (no "empathetic" trait), so empathy base is 0.5
    persona = make_expert(system)

    sad_behavior = system.adapt_persona_to_context(
        persona.persona_id, {"mood": "sad"}
    )
    assert sad_behavior["empathy"] == pytest.approx(0.8)  # 0.5 + 0.3

    complex_behavior = system.adapt_persona_to_context(
        persona.persona_id, {"complexity": "high"}
    )
    assert complex_behavior["verbosity"] == pytest.approx(0.8)  # 0.5 + 0.3


# ===========================================================================
# 8 – deactivate_persona
# ===========================================================================

def test_deactivate_persona_sets_active_false(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system)
    assert persona.active is True

    result = system.deactivate_persona(persona.persona_id)
    assert result is True
    assert persona.active is False

    # Persisted file should also reflect deactivation
    persona_file = tmp_path / "personas" / f"{persona.persona_id}.json"
    saved = json.loads(persona_file.read_text())
    assert saved["active"] is False


# ===========================================================================
# 9 – export_persona
# ===========================================================================

def test_export_persona_returns_valid_json(tmp_path):
    system = make_system(tmp_path)
    persona = make_expert(system, name="Export Me")

    exported = system.export_persona(persona.persona_id)
    assert isinstance(exported, str)

    data = json.loads(exported)  # must be valid JSON
    assert data["persona_id"] == persona.persona_id
    assert data["name"] == persona.name
    assert "traits" in data
    assert isinstance(data["traits"], dict)
