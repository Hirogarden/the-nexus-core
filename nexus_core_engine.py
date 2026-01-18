"""
The Nexus Core - Main RAG Engine
Handles hierarchical conversation logging, quality validation, and semantic search.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

# Optional dependencies - graceful degradation if not available
try:
    from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

try:
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class NexusCoreEngine:
    """
    Core RAG engine with hierarchical logging and quality validation.
    Works with zero dependencies but enhanced with LlamaIndex/LangChain if available.
    """
    
    def __init__(self, base_path: str = "./nexus_data"):
        """Initialize the Nexus Core engine."""
        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.index_path = self.base_path / "indices"
        self.sessions_path = self.base_path / "sessions"
        
        # Create directory structure
        self.conversations_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.sessions_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components based on available dependencies
        self.vector_index = None
        self.memory = None
        
        if LLAMA_INDEX_AVAILABLE:
            self._init_vector_index()
        
        if LANGCHAIN_AVAILABLE:
            self.memory = ConversationBufferMemory()
    
    def _init_vector_index(self):
        """Initialize or load vector index."""
        if not LLAMA_INDEX_AVAILABLE:
            return
        
        storage_path = self.index_path / "vector_store"
        
        try:
            if storage_path.exists():
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
                self.vector_index = load_index_from_storage(storage_context)
            else:
                self.vector_index = VectorStoreIndex([])
                storage_path.mkdir(parents=True, exist_ok=True)
                self.vector_index.storage_context.persist(persist_dir=str(storage_path))
        except Exception as e:
            print(f"Warning: Could not initialize vector index: {e}")
            self.vector_index = None
    
    def log_conversation_turn(
        self, 
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a conversation turn to hierarchical storage: Year/Month/Day/session.md
        
        Args:
            session_id: Unique session identifier
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Optional metadata (mode, quality_score, etc.)
        
        Returns:
            Dict with conversation turn info and file path
        """
        now = datetime.now()
        year_path = self.conversations_path / str(now.year)
        month_path = year_path / f"{now.month:02d}"
        day_path = month_path / f"{now.day:02d}"
        
        # Create hierarchical folders
        day_path.mkdir(parents=True, exist_ok=True)
        
        # Generate conversation file path
        conversation_file = day_path / f"{session_id}.md"
        
        # Prepare conversation entry
        timestamp = now.isoformat()
        quality_score = self.validate_conversation_quality(user_message, assistant_response)
        
        entry = {
            "timestamp": timestamp,
            "user": user_message,
            "assistant": assistant_response,
            "quality_score": quality_score,
            "metadata": metadata or {}
        }
        
        # Format as markdown
        md_entry = f"""
## {timestamp}
**Quality Score:** {quality_score:.2f}

### User
{user_message}

### Assistant
{assistant_response}

---
"""
        
        # Append to conversation file
        with open(conversation_file, "a", encoding="utf-8") as f:
            f.write(md_entry)
        
        # Update vector index if available
        if self.vector_index and LLAMA_INDEX_AVAILABLE:
            try:
                doc = Document(
                    text=f"User: {user_message}\nAssistant: {assistant_response}",
                    metadata={
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "quality_score": quality_score,
                        "file_path": str(conversation_file)
                    }
                )
                self.vector_index.insert(doc)
                self.vector_index.storage_context.persist(persist_dir=str(self.index_path / "vector_store"))
            except Exception as e:
                print(f"Warning: Could not update vector index: {e}")
        
        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "quality_score": quality_score,
            "file_path": str(conversation_file),
            "entry": entry
        }
    
    def validate_conversation_quality(self, user_message: str, assistant_response: str) -> float:
        """
        Validate conversation quality based on multiple signals.
        
        Returns a score between 0.0 and 1.0:
        - 1.0: High quality (good length, no repetition, coherent)
        - 0.5: Medium quality
        - 0.0: Low quality (too short, repetitive, incoherent)
        """
        score = 1.0
        
        # Check message length
        if len(user_message) < 5:
            score -= 0.2
        if len(assistant_response) < 10:
            score -= 0.3
        
        # Check for excessive repetition
        user_words = user_message.lower().split()
        assistant_words = assistant_response.lower().split()
        
        if len(set(user_words)) < len(user_words) * 0.5:
            score -= 0.2
        if len(set(assistant_words)) < len(assistant_words) * 0.5:
            score -= 0.2
        
        # Check for coherence (basic heuristic: presence of punctuation and complete sentences)
        if "." not in assistant_response and "!" not in assistant_response and "?" not in assistant_response:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        time_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across conversations.
        
        Args:
            query: Search query
            top_k: Number of results to return
            time_filter: Optional time filter {"year": 2024, "month": 1, "day": 15}
        
        Returns:
            List of relevant conversation turns with metadata
        """
        results = []
        
        if self.vector_index and LLAMA_INDEX_AVAILABLE:
            try:
                # Use vector search if available
                query_engine = self.vector_index.as_query_engine(similarity_top_k=top_k)
                response = query_engine.query(query)
                
                for node in response.source_nodes:
                    results.append({
                        "text": node.text,
                        "score": node.score,
                        "metadata": node.metadata
                    })
            except Exception as e:
                print(f"Warning: Vector search failed: {e}")
        
        # Fallback: keyword-based search through conversation files
        if not results:
            results = self._keyword_search(query, top_k, time_filter)
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
        time_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        results = []
        query_lower = query.lower()
        
        # Determine search path based on time filter
        if time_filter:
            year = time_filter.get("year")
            month = time_filter.get("month")
            day = time_filter.get("day")
            
            search_path = self.conversations_path
            if year:
                search_path = search_path / str(year)
            if month:
                search_path = search_path / f"{month:02d}"
            if day:
                search_path = search_path / f"{day:02d}"
        else:
            search_path = self.conversations_path
        
        # Search through files
        if search_path.exists():
            for file_path in search_path.rglob("*.md"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if query_lower in content.lower():
                        # Count occurrences for simple relevance scoring
                        score = content.lower().count(query_lower) / len(content.split())
                        
                        results.append({
                            "text": content[:500] + "...",  # Preview
                            "score": score,
                            "metadata": {
                                "file_path": str(file_path),
                                "session_id": file_path.stem
                            }
                        })
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific session."""
        # Search for session file
        for file_path in self.conversations_path.rglob(f"{session_id}.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                
                # Parse markdown to extract entries
                entries = content.split("---")
                
                return {
                    "session_id": session_id,
                    "file_path": str(file_path),
                    "entry_count": len([e for e in entries if e.strip()]),
                    "content_preview": content[:300] + "..."
                }
            except Exception as e:
                print(f"Warning: Could not read session file: {e}")
        
        return None
    
    def export_session(self, session_id: str, output_format: str = "json") -> Optional[str]:
        """Export a session in specified format (json, md, txt)."""
        session_info = self.get_session_summary(session_id)
        
        if not session_info:
            return None
        
        file_path = Path(session_info["file_path"])
        content = file_path.read_text(encoding="utf-8")
        
        if output_format == "json":
            # Convert markdown to JSON structure
            export_data = {
                "session_id": session_id,
                "file_path": str(file_path),
                "raw_content": content
            }
            return json.dumps(export_data, indent=2)
        elif output_format == "md":
            return content
        elif output_format == "txt":
            # Strip markdown formatting
            return content.replace("#", "").replace("**", "").replace("---", "\n")
        else:
            return content


if __name__ == "__main__":
    # Demo usage
    engine = NexusCoreEngine("./demo_nexus_data")
    
    print("🌟 Nexus Core Engine initialized")
    print(f"Vector index available: {LLAMA_INDEX_AVAILABLE}")
    print(f"Memory management available: {LANGCHAIN_AVAILABLE}")
    
    # Log a demo conversation
    result = engine.log_conversation_turn(
        session_id="demo_session",
        user_message="What is the purpose of The Nexus Core?",
        assistant_response="The Nexus Core is an advanced RAG system with hierarchical indexing, quality validation, and intelligent search capabilities.",
        metadata={"mode": "companion"}
    )
    
    print(f"\n✅ Logged conversation to: {result['file_path']}")
    print(f"Quality score: {result['quality_score']:.2f}")
    
    # Perform a search
    results = engine.semantic_search("purpose of Nexus Core", top_k=3)
    print(f"\n🔍 Search results: {len(results)} found")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f} - {result['text'][:100]}...")
