"""
The Nexus Core - Layered Memory System
Brain-like memory with short-term (working) and long-term (persistent) memory layers.
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
from collections import deque


@dataclass
class MemoryItem:
    """Represents a single memory item."""
    memory_id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural"
    importance: float  # 0.0 to 1.0
    access_count: int
    created_at: str
    last_accessed: str
    decay_rate: float  # How quickly importance decays
    tags: List[str]
    context: Dict[str, Any]


@dataclass
class ConsolidationRule:
    """Rules for memory consolidation from short-term to long-term."""
    min_access_count: int
    min_importance: float
    min_age_hours: float
    required_tags: Optional[List[str]] = None


class ShortTermMemory:
    """
    Working memory - fast, limited capacity, temporary storage.
    Similar to human short-term memory.
    """
    
    def __init__(self, max_capacity: int = 20, retention_minutes: int = 30):
        """
        Initialize short-term memory.
        
        Args:
            max_capacity: Maximum number of items in short-term memory
            retention_minutes: How long items stay before decay
        """
        self.max_capacity = max_capacity
        self.retention_minutes = retention_minutes
        self.items: deque = deque(maxlen=max_capacity)
        self.item_map: Dict[str, MemoryItem] = {}
    
    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """Store an item in short-term memory."""
        memory_id = hashlib.sha256(
            f"{content}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        item = MemoryItem(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            access_count=1,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            decay_rate=0.1,
            tags=tags or [],
            context=context or {}
        )
        
        # Add to deque (automatically removes oldest if at capacity)
        self.items.append(item)
        self.item_map[memory_id] = item
        
        # Clean up old items from map
        self._cleanup_expired()
        
        return item
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve an item from short-term memory."""
        item = self.item_map.get(memory_id)
        if item:
            item.access_count += 1
            item.last_accessed = datetime.now().isoformat()
            # Increase importance with access
            item.importance = min(1.0, item.importance + 0.05)
        return item
    
    def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Search short-term memory."""
        query_lower = query.lower()
        matches = []
        
        for item in self.items:
            if query_lower in item.content.lower():
                relevance = item.content.lower().count(query_lower) / len(item.content.split())
                matches.append((item, relevance * item.importance))
        
        # Sort by relevance
        matches.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in matches[:top_k]]
    
    def _cleanup_expired(self):
        """Remove expired items from short-term memory."""
        current_time = datetime.now()
        retention_delta = timedelta(minutes=self.retention_minutes)
        
        expired_ids = []
        for memory_id, item in self.item_map.items():
            created_time = datetime.fromisoformat(item.created_at)
            if current_time - created_time > retention_delta:
                expired_ids.append(memory_id)
        
        for memory_id in expired_ids:
            del self.item_map[memory_id]
    
    def get_all(self) -> List[MemoryItem]:
        """Get all items in short-term memory."""
        return list(self.items)
    
    def get_important_items(self, threshold: float = 0.7) -> List[MemoryItem]:
        """Get items above importance threshold."""
        return [item for item in self.items if item.importance >= threshold]
    
    def clear(self):
        """Clear all short-term memory."""
        self.items.clear()
        self.item_map.clear()


class LongTermMemory:
    """
    Persistent memory - large capacity, durable storage.
    Similar to human long-term memory.
    """
    
    def __init__(self, storage_path: str = "./nexus_data/long_term_memory"):
        """Initialize long-term memory with persistent storage."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for fast access
        self.index: Dict[str, MemoryItem] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[str, Set[str]] = {}
        
        # Load existing memories
        self._load_memories()
    
    def _load_memories(self):
        """Load memories from persistent storage."""
        for memory_file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(memory_file.read_text())
                item = MemoryItem(**data)
                self.index[item.memory_id] = item
                
                # Update indices
                for tag in item.tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = set()
                    self.tag_index[tag].add(item.memory_id)
                
                if item.memory_type not in self.type_index:
                    self.type_index[item.memory_type] = set()
                self.type_index[item.memory_type].add(item.memory_id)
                
            except Exception as e:
                print(f"Warning: Could not load memory from {memory_file}: {e}")
    
    def _save_memory(self, item: MemoryItem):
        """Save a memory to persistent storage."""
        memory_file = self.storage_path / f"{item.memory_id}.json"
        try:
            memory_file.write_text(json.dumps(asdict(item), indent=2))
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")
    
    def store(self, item: MemoryItem) -> bool:
        """Store an item in long-term memory."""
        self.index[item.memory_id] = item
        self._save_memory(item)
        
        # Update indices
        for tag in item.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(item.memory_id)
        
        if item.memory_type not in self.type_index:
            self.type_index[item.memory_type] = set()
        self.type_index[item.memory_type].add(item.memory_id)
        
        return True
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve an item from long-term memory."""
        item = self.index.get(memory_id)
        if item:
            item.access_count += 1
            item.last_accessed = datetime.now().isoformat()
            self._save_memory(item)
        return item
    
    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        top_k: int = 10
    ) -> List[MemoryItem]:
        """Search long-term memory with multiple filters."""
        candidates = set(self.index.keys())
        
        # Filter by tags
        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self.tag_index:
                    tag_matches.update(self.tag_index[tag])
            candidates &= tag_matches
        
        # Filter by type
        if memory_type and memory_type in self.type_index:
            candidates &= self.type_index[memory_type]
        
        # Get items and filter by importance
        items = [self.index[mid] for mid in candidates if mid in self.index]
        items = [item for item in items if item.importance >= min_importance]
        
        # Filter by query
        if query:
            query_lower = query.lower()
            scored_items = []
            for item in items:
                if query_lower in item.content.lower():
                    relevance = item.content.lower().count(query_lower) / len(item.content.split())
                    score = relevance * item.importance * (1 + item.access_count * 0.1)
                    scored_items.append((item, score))
            
            scored_items.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in scored_items[:top_k]]
        
        # Sort by importance and recency
        items.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return items[:top_k]
    
    def prune(self, min_importance: float = 0.1, max_age_days: int = 365):
        """Remove low-importance or very old memories."""
        current_time = datetime.now()
        max_age = timedelta(days=max_age_days)
        
        # Days in a year for decay calculation
        DAYS_PER_YEAR = 365.0
        
        to_remove = []
        for memory_id, item in self.index.items():
            created_time = datetime.fromisoformat(item.created_at)
            age = current_time - created_time
            
            # Apply decay
            decayed_importance = item.importance * (1 - item.decay_rate * (age.days / DAYS_PER_YEAR))
            
            if decayed_importance < min_importance or age > max_age:
                to_remove.append(memory_id)
        
        # Remove memories
        for memory_id in to_remove:
            item = self.index[memory_id]
            del self.index[memory_id]
            
            # Remove from indices
            for tag in item.tags:
                if tag in self.tag_index:
                    self.tag_index[tag].discard(memory_id)
            
            if item.memory_type in self.type_index:
                self.type_index[item.memory_type].discard(memory_id)
            
            # Remove file
            memory_file = self.storage_path / f"{memory_id}.json"
            if memory_file.exists():
                memory_file.unlink()
        
        return len(to_remove)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about long-term memory."""
        if not self.index:
            return {"total_memories": 0}
        
        items = list(self.index.values())
        
        return {
            "total_memories": len(items),
            "avg_importance": sum(i.importance for i in items) / len(items),
            "avg_access_count": sum(i.access_count for i in items) / len(items),
            "memory_types": {k: len(v) for k, v in self.type_index.items()},
            "top_tags": sorted(
                [(tag, len(ids)) for tag, ids in self.tag_index.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class LayeredMemorySystem:
    """
    Complete layered memory system combining short-term and long-term memory.
    Automatically consolidates important short-term memories into long-term storage.
    """
    
    def __init__(
        self,
        stm_capacity: int = 20,
        stm_retention_minutes: int = 30,
        ltm_storage_path: str = "./nexus_data/long_term_memory"
    ):
        """Initialize the layered memory system."""
        self.short_term = ShortTermMemory(stm_capacity, stm_retention_minutes)
        self.long_term = LongTermMemory(ltm_storage_path)
        
        # Default consolidation rules
        self.consolidation_rules = [
            ConsolidationRule(
                min_access_count=2,
                min_importance=0.7,
                min_age_hours=0.5
            ),
            ConsolidationRule(
                min_access_count=5,
                min_importance=0.5,
                min_age_hours=1.0
            )
        ]
    
    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        force_long_term: bool = False
    ) -> MemoryItem:
        """
        Store a memory item.
        
        Args:
            content: Memory content
            memory_type: Type of memory (episodic, semantic, procedural)
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization
            context: Optional context information
            force_long_term: If True, store directly in long-term memory
        
        Returns:
            Created MemoryItem
        """
        item = self.short_term.store(content, memory_type, importance, tags, context)
        
        if force_long_term or importance >= 0.8:
            # Store directly in long-term for very important items
            self.long_term.store(item)
        
        return item
    
    def retrieve(self, query: str, search_both: bool = True) -> List[MemoryItem]:
        """
        Retrieve memories matching a query.
        
        Args:
            query: Search query
            search_both: If True, search both short and long-term memory
        
        Returns:
            List of matching memories
        """
        results = []
        
        # Search short-term memory
        stm_results = self.short_term.search(query, top_k=5)
        results.extend(stm_results)
        
        # Search long-term memory
        if search_both:
            ltm_results = self.long_term.search(query=query, top_k=5)
            results.extend(ltm_results)
        
        # Remove duplicates and sort by importance
        seen_ids = set()
        unique_results = []
        for item in results:
            if item.memory_id not in seen_ids:
                seen_ids.add(item.memory_id)
                unique_results.append(item)
        
        unique_results.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return unique_results
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Consolidate memories from short-term to long-term based on consolidation rules.
        
        Returns:
            Statistics about consolidation
        """
        consolidated_count = 0
        current_time = datetime.now()
        
        for item in self.short_term.get_all():
            created_time = datetime.fromisoformat(item.created_at)
            age_hours = (current_time - created_time).total_seconds() / 3600
            
            # Check against consolidation rules
            for rule in self.consolidation_rules:
                if (item.access_count >= rule.min_access_count and
                    item.importance >= rule.min_importance and
                    age_hours >= rule.min_age_hours):
                    
                    # Check required tags if specified
                    if rule.required_tags:
                        if not any(tag in item.tags for tag in rule.required_tags):
                            continue
                    
                    # Consolidate to long-term memory
                    if item.memory_id not in self.long_term.index:
                        self.long_term.store(item)
                        consolidated_count += 1
                    break
        
        return {
            "consolidated": consolidated_count,
            "stm_count": len(self.short_term.get_all()),
            "ltm_count": len(self.long_term.index),
            "timestamp": current_time.isoformat()
        }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get status of the entire memory system."""
        stm_items = self.short_term.get_all()
        ltm_stats = self.long_term.get_statistics()
        
        return {
            "short_term": {
                "count": len(stm_items),
                "capacity": self.short_term.max_capacity,
                "utilization": len(stm_items) / self.short_term.max_capacity,
                "avg_importance": sum(i.importance for i in stm_items) / len(stm_items) if stm_items else 0
            },
            "long_term": ltm_stats,
            "consolidation_rules": len(self.consolidation_rules)
        }
    
    def auto_maintain(self):
        """Perform automatic maintenance on memory system."""
        # Consolidate memories
        self.consolidate_memories()
        
        # Prune old, unimportant long-term memories
        self.long_term.prune(min_importance=0.1, max_age_days=365)


if __name__ == "__main__":
    # Demo usage
    print("🧠 Layered Memory System - Demo\n")
    
    # Initialize memory system
    memory = LayeredMemorySystem(
        stm_capacity=10,
        stm_retention_minutes=30,
        ltm_storage_path="./demo_memory"
    )
    
    print("✅ Memory system initialized")
    
    # Store various memories
    print("\n💾 Storing memories...")
    
    memory.store(
        "Machine learning is a subset of AI",
        memory_type="semantic",
        importance=0.8,
        tags=["AI", "machine learning", "education"]
    )
    
    memory.store(
        "Had a great conversation about neural networks today",
        memory_type="episodic",
        importance=0.6,
        tags=["conversation", "neural networks"]
    )
    
    memory.store(
        "To implement backpropagation: 1) forward pass, 2) calculate loss, 3) backward pass",
        memory_type="procedural",
        importance=0.9,
        tags=["algorithm", "neural networks", "implementation"]
    )
    
    print("   Stored 3 memories")
    
    # Retrieve memories
    print("\n🔍 Retrieving memories about 'neural networks'...")
    results = memory.retrieve("neural networks")
    for i, item in enumerate(results, 1):
        print(f"   {i}. [{item.memory_type}] {item.content[:60]}... (importance: {item.importance:.2f})")
    
    # Consolidate
    print("\n🔄 Consolidating memories...")
    consolidation_result = memory.consolidate_memories()
    print(f"   Consolidated: {consolidation_result['consolidated']}")
    print(f"   STM: {consolidation_result['stm_count']}, LTM: {consolidation_result['ltm_count']}")
    
    # Show status
    print("\n📊 Memory System Status:")
    status = memory.get_memory_status()
    print(f"   STM utilization: {status['short_term']['utilization']:.1%}")
    print(f"   LTM total: {status['long_term']['total_memories']}")
