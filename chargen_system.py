"""
The Nexus Core - Character/Persona Generation System
Dynamic persona creation and management for context-aware interactions.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import hashlib


@dataclass
class PersonaTrait:
    """Represents a personality trait."""
    name: str
    value: float  # 0.0 to 1.0
    description: str


@dataclass
class Persona:
    """Represents a complete character/persona."""
    persona_id: str
    name: str
    role: str
    traits: Dict[str, PersonaTrait]
    knowledge_domains: List[str]
    communication_style: str
    goals: List[str]
    backstory: Optional[str] = None
    active: bool = True
    interaction_count: int = 0
    created_at: str = None
    last_used: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_used is None:
            self.last_used = datetime.now().isoformat()


class CharGenSystem:
    """
    Character/Persona generation and management system.
    Creates and maintains distinct personas for different interaction contexts.
    """
    
    def __init__(self, storage_path: str = "./nexus_data/personas"):
        """Initialize the CharGen system."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.personas: Dict[str, Persona] = {}
        self.persona_templates: Dict[str, Dict[str, Any]] = {}
        
        # Load existing personas
        self._load_personas()
        
        # Register default templates
        self._register_default_templates()
    
    def _load_personas(self):
        """Load personas from storage."""
        for persona_file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(persona_file.read_text())
                # Reconstruct traits
                traits = {}
                for trait_name, trait_data in data.get("traits", {}).items():
                    traits[trait_name] = PersonaTrait(**trait_data)
                data["traits"] = traits
                
                persona = Persona(**data)
                self.personas[persona.persona_id] = persona
            except Exception as e:
                print(f"Warning: Could not load persona from {persona_file}: {e}")
    
    def _save_persona(self, persona: Persona):
        """Save a persona to storage."""
        persona_file = self.storage_path / f"{persona.persona_id}.json"
        
        # Convert to dict for JSON serialization
        data = asdict(persona)
        # Convert trait objects to dicts
        data["traits"] = {
            name: asdict(trait) for name, trait in persona.traits.items()
        }
        
        try:
            persona_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Warning: Could not save persona: {e}")
    
    def _register_default_templates(self):
        """Register default persona templates."""
        self.persona_templates = {
            "expert": {
                "role": "domain expert",
                "traits": {
                    "analytical": PersonaTrait("analytical", 0.9, "Strong analytical thinking"),
                    "confident": PersonaTrait("confident", 0.85, "High confidence in expertise"),
                    "formal": PersonaTrait("formal", 0.7, "Formal communication style")
                },
                "communication_style": "precise, technical, authoritative",
                "goals": ["provide accurate information", "educate users", "maintain credibility"]
            },
            "companion": {
                "role": "friendly companion",
                "traits": {
                    "empathetic": PersonaTrait("empathetic", 0.95, "Highly empathetic"),
                    "supportive": PersonaTrait("supportive", 0.9, "Very supportive"),
                    "casual": PersonaTrait("casual", 0.8, "Casual communication")
                },
                "communication_style": "warm, conversational, encouraging",
                "goals": ["provide emotional support", "engage meaningfully", "build rapport"]
            },
            "analyst": {
                "role": "data analyst",
                "traits": {
                    "logical": PersonaTrait("logical", 0.95, "Highly logical reasoning"),
                    "detail_oriented": PersonaTrait("detail_oriented", 0.9, "Strong attention to detail"),
                    "objective": PersonaTrait("objective", 0.85, "Objective perspective")
                },
                "communication_style": "structured, data-driven, methodical",
                "goals": ["analyze patterns", "provide insights", "ensure accuracy"]
            },
            "creative": {
                "role": "creative thinker",
                "traits": {
                    "imaginative": PersonaTrait("imaginative", 0.95, "Highly imaginative"),
                    "flexible": PersonaTrait("flexible", 0.85, "Flexible thinking"),
                    "expressive": PersonaTrait("expressive", 0.9, "Expressive communication")
                },
                "communication_style": "vivid, metaphorical, exploratory",
                "goals": ["generate novel ideas", "think outside the box", "inspire creativity"]
            },
            "teacher": {
                "role": "educator",
                "traits": {
                    "patient": PersonaTrait("patient", 0.95, "Very patient"),
                    "clear": PersonaTrait("clear", 0.9, "Clear communication"),
                    "encouraging": PersonaTrait("encouraging", 0.85, "Encouraging approach")
                },
                "communication_style": "pedagogical, step-by-step, supportive",
                "goals": ["facilitate learning", "break down complexity", "build understanding"]
            }
        }
    
    def generate_persona(
        self,
        name: str,
        template: str = "expert",
        knowledge_domains: Optional[List[str]] = None,
        custom_traits: Optional[Dict[str, PersonaTrait]] = None,
        backstory: Optional[str] = None
    ) -> Persona:
        """
        Generate a new persona.
        
        Args:
            name: Persona name
            template: Template to use (expert, companion, analyst, creative, teacher)
            knowledge_domains: List of knowledge domains
            custom_traits: Optional custom traits to override/add
            backstory: Optional backstory
        
        Returns:
            Generated Persona object
        """
        if template not in self.persona_templates:
            template = "expert"
        
        template_data = self.persona_templates[template]
        
        # Generate unique ID using secure hash
        persona_id = hashlib.sha256(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Combine traits
        traits = template_data["traits"].copy()
        if custom_traits:
            traits.update(custom_traits)
        
        # Create persona
        persona = Persona(
            persona_id=persona_id,
            name=name,
            role=template_data["role"],
            traits=traits,
            knowledge_domains=knowledge_domains or [],
            communication_style=template_data["communication_style"],
            goals=template_data["goals"],
            backstory=backstory
        )
        
        # Store persona
        self.personas[persona_id] = persona
        self._save_persona(persona)
        
        return persona
    
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Retrieve a persona by ID."""
        return self.personas.get(persona_id)
    
    def list_personas(
        self,
        active_only: bool = True,
        role_filter: Optional[str] = None
    ) -> List[Persona]:
        """
        List available personas.
        
        Args:
            active_only: Only return active personas
            role_filter: Optional role filter
        
        Returns:
            List of matching personas
        """
        personas = list(self.personas.values())
        
        if active_only:
            personas = [p for p in personas if p.active]
        
        if role_filter:
            personas = [p for p in personas if role_filter.lower() in p.role.lower()]
        
        return personas
    
    def update_persona_traits(
        self,
        persona_id: str,
        trait_updates: Dict[str, float]
    ) -> bool:
        """
        Update persona traits based on interactions.
        
        Args:
            persona_id: Persona to update
            trait_updates: Dict of trait names and value changes (-1.0 to 1.0)
        
        Returns:
            True if successful
        """
        persona = self.personas.get(persona_id)
        if not persona:
            return False
        
        for trait_name, change in trait_updates.items():
            if trait_name in persona.traits:
                trait = persona.traits[trait_name]
                new_value = max(0.0, min(1.0, trait.value + change))
                trait.value = new_value
        
        self._save_persona(persona)
        return True
    
    def record_interaction(self, persona_id: str):
        """Record an interaction with a persona."""
        persona = self.personas.get(persona_id)
        if persona:
            persona.interaction_count += 1
            persona.last_used = datetime.now().isoformat()
            self._save_persona(persona)
    
    def get_persona_profile(self, persona_id: str) -> Optional[str]:
        """
        Get a formatted profile for a persona.
        
        Args:
            persona_id: Persona ID
        
        Returns:
            Formatted profile string
        """
        persona = self.personas.get(persona_id)
        if not persona:
            return None
        
        lines = [
            f"# Persona Profile: {persona.name}",
            f"**ID:** {persona.persona_id}",
            f"**Role:** {persona.role}",
            f"**Status:** {'Active' if persona.active else 'Inactive'}",
            "",
            "## Personality Traits",
        ]
        
        for trait_name, trait in persona.traits.items():
            bar = "█" * int(trait.value * 10) + "░" * (10 - int(trait.value * 10))
            lines.append(f"- **{trait_name.title()}**: {bar} {trait.value:.2f}")
            lines.append(f"  _{trait.description}_")
        
        lines.extend([
            "",
            "## Knowledge Domains",
        ])
        
        for domain in persona.knowledge_domains:
            lines.append(f"- {domain}")
        
        lines.extend([
            "",
            f"**Communication Style:** {persona.communication_style}",
            "",
            "## Goals",
        ])
        
        for goal in persona.goals:
            lines.append(f"- {goal}")
        
        if persona.backstory:
            lines.extend([
                "",
                "## Backstory",
                persona.backstory
            ])
        
        lines.extend([
            "",
            "## Statistics",
            f"- Created: {persona.created_at}",
            f"- Last used: {persona.last_used}",
            f"- Interactions: {persona.interaction_count}",
        ])
        
        return "\n".join(lines)
    
    def adapt_persona_to_context(
        self,
        persona_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt persona behavior based on context.
        
        Args:
            persona_id: Persona to adapt
            context: Context information (mood, topic, urgency, etc.)
        
        Returns:
            Adapted behavior parameters
        """
        persona = self.personas.get(persona_id)
        if not persona:
            return {}
        
        # Base behavior from persona
        behavior = {
            "communication_style": persona.communication_style,
            "formality": 0.5,
            "verbosity": 0.5,
            "empathy": 0.5
        }
        
        # Adjust based on traits
        if "formal" in persona.traits:
            behavior["formality"] = persona.traits["formal"].value
        if "empathetic" in persona.traits:
            behavior["empathy"] = persona.traits["empathetic"].value
        
        # Adapt to context
        if context.get("urgency") == "high":
            behavior["verbosity"] = max(0.0, behavior["verbosity"] - 0.3)
            behavior["formality"] = min(1.0, behavior["formality"] + 0.2)
        
        if context.get("mood") == "sad":
            behavior["empathy"] = min(1.0, behavior["empathy"] + 0.3)
        
        if context.get("complexity") == "high":
            behavior["verbosity"] = min(1.0, behavior["verbosity"] + 0.3)
        
        return behavior
    
    def deactivate_persona(self, persona_id: str) -> bool:
        """Deactivate a persona."""
        persona = self.personas.get(persona_id)
        if persona:
            persona.active = False
            self._save_persona(persona)
            return True
        return False
    
    def export_persona(self, persona_id: str) -> Optional[str]:
        """Export a persona as JSON."""
        persona = self.personas.get(persona_id)
        if not persona:
            return None
        
        data = asdict(persona)
        data["traits"] = {
            name: asdict(trait) for name, trait in persona.traits.items()
        }
        
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    # Demo usage
    print("👤 CharGen System - Demo\n")
    
    # Initialize system
    chargen = CharGenSystem("./demo_personas")
    print("✅ CharGen system initialized")
    
    # Generate different personas
    print("\n🎭 Generating personas...")
    
    expert = chargen.generate_persona(
        name="Dr. Sarah Chen",
        template="expert",
        knowledge_domains=["machine learning", "neural networks", "data science"],
        backstory="PhD in Computer Science, 10 years of industry experience"
    )
    print(f"   Created expert: {expert.name} ({expert.persona_id})")
    
    companion = chargen.generate_persona(
        name="Alex",
        template="companion",
        knowledge_domains=["general conversation", "emotional support"],
        backstory="Friendly AI companion focused on meaningful interactions"
    )
    print(f"   Created companion: {companion.name} ({companion.persona_id})")
    
    # Show persona profiles
    print("\n📋 Expert Profile:")
    print(chargen.get_persona_profile(expert.persona_id))
    
    # Demonstrate context adaptation
    print("\n🔄 Adapting to different contexts...")
    
    urgent_context = {"urgency": "high", "topic": "bug_fix"}
    behavior = chargen.adapt_persona_to_context(expert.persona_id, urgent_context)
    print(f"   Urgent situation: formality={behavior['formality']:.2f}, verbosity={behavior['verbosity']:.2f}")
    
    emotional_context = {"mood": "sad", "topic": "personal"}
    behavior = chargen.adapt_persona_to_context(companion.persona_id, emotional_context)
    print(f"   Emotional situation: empathy={behavior['empathy']:.2f}")
    
    # Record interactions
    chargen.record_interaction(expert.persona_id)
    chargen.record_interaction(expert.persona_id)
    chargen.record_interaction(companion.persona_id)
    
    # List all personas
    print(f"\n📊 Total personas: {len(chargen.list_personas())}")
