"""
The Nexus Core - Advanced Enhancement Features
Addresses 7 common RAG system complaints with production-ready solutions.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import hashlib
import re
from collections import defaultdict, Counter


class CitationManager:
    """
    Feature 1: Citation Tracking
    Track and format source attributions for every response.
    """
    
    def __init__(self):
        self.citation_map: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_citation(
        self,
        response_id: str,
        source_id: str,
        source_type: str,
        relevance_score: float,
        excerpt: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a citation for a response."""
        if response_id not in self.citation_map:
            self.citation_map[response_id] = []
        
        citation = {
            "source_id": source_id,
            "source_type": source_type,
            "relevance_score": relevance_score,
            "excerpt": excerpt,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.citation_map[response_id].append(citation)
    
    def format_citations(
        self,
        response_id: str,
        format_style: str = "numbered"
    ) -> str:
        """
        Format citations for display.
        
        Styles: "numbered", "inline", "footnote", "apa"
        """
        if response_id not in self.citation_map:
            return ""
        
        citations = self.citation_map[response_id]
        
        if format_style == "numbered":
            lines = ["**Sources:**"]
            for i, cit in enumerate(citations, 1):
                lines.append(f"{i}. {cit['source_type']} ({cit['source_id']}) - Relevance: {cit['relevance_score']:.2f}")
                lines.append(f"   \"{cit['excerpt'][:100]}...\"")
            return "\n".join(lines)
        
        elif format_style == "inline":
            parts = []
            for cit in citations:
                parts.append(f"[{cit['source_id']}]")
            return " ".join(parts)
        
        elif format_style == "footnote":
            lines = []
            for i, cit in enumerate(citations, 1):
                lines.append(f"[^{i}]: {cit['source_type']} - {cit['source_id']}")
            return "\n".join(lines)
        
        elif format_style == "apa":
            lines = ["**References:**"]
            for cit in citations:
                meta = cit.get("metadata", {})
                author = meta.get("author", "Unknown")
                year = meta.get("year", "n.d.")
                lines.append(f"{author} ({year}). {cit['source_id']}. Retrieved from {cit['source_type']}")
            return "\n".join(lines)
        
        return ""
    
    def get_citation_quality(self, response_id: str) -> Dict[str, Any]:
        """Assess citation quality for a response."""
        if response_id not in self.citation_map:
            return {"quality": "none", "count": 0}
        
        citations = self.citation_map[response_id]
        
        avg_relevance = sum(c["relevance_score"] for c in citations) / len(citations)
        unique_sources = len(set(c["source_id"] for c in citations))
        
        quality = "poor"
        if avg_relevance > 0.8 and unique_sources >= 3:
            quality = "excellent"
        elif avg_relevance > 0.6 and unique_sources >= 2:
            quality = "good"
        elif unique_sources >= 1:
            quality = "fair"
        
        return {
            "quality": quality,
            "count": len(citations),
            "unique_sources": unique_sources,
            "avg_relevance": avg_relevance
        }


class ContextWindowManager:
    """
    Feature 2: Context Window Management
    Smart compression and prioritization for long conversations.
    """
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
    
    def compress_context(
        self,
        messages: List[Dict[str, str]],
        priorities: Optional[List[float]] = None
    ) -> List[Dict[str, str]]:
        """
        Compress context to fit within token limit.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            priorities: Optional priority scores for each message (0-1)
        
        Returns:
            Compressed message list
        """
        if not messages:
            return []
        
        # Estimate tokens (rough: 4 chars = 1 token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in messages)
        
        if total_tokens <= self.max_tokens:
            return messages
        
        # Apply compression strategy
        if priorities:
            return self._priority_based_compression(messages, priorities)
        else:
            return self._recency_based_compression(messages)
    
    def _priority_based_compression(
        self,
        messages: List[Dict[str, str]],
        priorities: List[float]
    ) -> List[Dict[str, str]]:
        """Keep highest priority messages."""
        # Sort by priority (descending)
        indexed_messages = list(zip(messages, priorities, range(len(messages))))
        indexed_messages.sort(key=lambda x: x[1], reverse=True)
        
        compressed = []
        token_count = 0
        
        for msg, priority, original_idx in indexed_messages:
            msg_tokens = len(msg["content"]) // 4
            if token_count + msg_tokens <= self.max_tokens:
                compressed.append((msg, original_idx))
                token_count += msg_tokens
        
        # Restore original order
        compressed.sort(key=lambda x: x[1])
        return [msg for msg, _ in compressed]
    
    def _recency_based_compression(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Keep most recent messages."""
        compressed = []
        token_count = 0
        
        # Start from most recent
        for msg in reversed(messages):
            msg_tokens = len(msg["content"]) // 4
            if token_count + msg_tokens <= self.max_tokens:
                compressed.insert(0, msg)
                token_count += msg_tokens
            else:
                break
        
        return compressed
    
    def summarize_dropped_context(
        self,
        original: List[Dict[str, str]],
        compressed: List[Dict[str, str]]
    ) -> str:
        """Generate summary of dropped messages."""
        dropped_count = len(original) - len(compressed)
        
        if dropped_count == 0:
            return ""
        
        return f"[{dropped_count} earlier messages summarized for context]"


class DeduplicationEngine:
    """
    Feature 3: Deduplication
    Remove duplicate or near-duplicate search results.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicates from search results.
        
        Methods: "hash", "semantic", "hybrid"
        """
        if method == "hash":
            return self._hash_based_dedup(results)
        elif method == "semantic":
            return self._semantic_dedup(results)
        else:  # hybrid
            # First hash-based for exact matches
            results = self._hash_based_dedup(results)
            # Then semantic for near-duplicates
            return self._semantic_dedup(results)
    
    def _hash_based_dedup(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove exact duplicates using content hashing."""
        seen_hashes: Set[str] = set()
        deduped = []
        
        for result in results:
            content = result.get("text", "")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduped.append(result)
        
        return deduped
    
    def _semantic_dedup(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicates using semantic similarity."""
        deduped = []
        
        for result in results:
            is_duplicate = False
            
            for existing in deduped:
                similarity = self._calculate_similarity(
                    result.get("text", ""),
                    existing.get("text", "")
                )
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduped.append(result)
        
        return deduped
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        # Tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0


class RelevanceRanker:
    """
    Feature 4: Result Re-ranking
    Multi-signal relevance scoring with user feedback incorporation.
    """
    
    def __init__(self):
        self.feedback_history: Dict[str, List[float]] = defaultdict(list)
    
    def rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using multiple signals.
        
        Signals:
        - Original relevance score (40%)
        - Recency (20%)
        - Quality score (20%)
        - Session context (10%)
        - User feedback (10%)
        """
        if not results:
            return []
        
        reranked = []
        
        for result in results:
            # Calculate combined score
            original_score = result.get("score", 0.5)
            recency_score = self._calculate_recency(result.get("metadata", {}))
            quality_score = result.get("metadata", {}).get("quality_score", 0.7)
            context_score = self._calculate_context_match(result, user_context)
            feedback_score = self._get_feedback_score(result.get("metadata", {}).get("doc_id"))
            
            combined_score = (
                original_score * 0.4 +
                recency_score * 0.2 +
                quality_score * 0.2 +
                context_score * 0.1 +
                feedback_score * 0.1
            )
            
            result["reranked_score"] = combined_score
            reranked.append(result)
        
        # Sort by reranked score
        reranked.sort(key=lambda x: x["reranked_score"], reverse=True)
        
        return reranked
    
    def _calculate_recency(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score (newer = higher)."""
        timestamp_str = metadata.get("timestamp")
        if not timestamp_str:
            return 0.5
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.now() - timestamp).days
            
            # Exponential decay: 0.99^days
            return 0.99 ** age_days
        except:
            return 0.5
    
    def _calculate_context_match(
        self,
        result: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate how well result matches user context."""
        if not user_context:
            return 0.5
        
        score = 0.5
        metadata = result.get("metadata", {})
        
        # Check mode match
        if user_context.get("mode") == metadata.get("mode"):
            score += 0.3
        
        # Check topic overlap
        user_topics = set(user_context.get("topics", []))
        result_topics = set(metadata.get("topics", []))
        
        if user_topics and result_topics:
            overlap = len(user_topics & result_topics) / len(user_topics | result_topics)
            score += overlap * 0.2
        
        return min(1.0, score)
    
    def _get_feedback_score(self, doc_id: Optional[str]) -> float:
        """Get average user feedback for a document."""
        if not doc_id or doc_id not in self.feedback_history:
            return 0.5
        
        feedback_list = self.feedback_history[doc_id]
        return sum(feedback_list) / len(feedback_list)
    
    def record_feedback(self, doc_id: str, score: float):
        """Record user feedback (0-1) for a document."""
        self.feedback_history[doc_id].append(max(0.0, min(1.0, score)))


class ConversationThreadTracker:
    """
    Feature 5: Thread Tracking
    Detect and maintain separate conversation topics.
    """
    
    def __init__(self, topic_change_threshold: float = 0.5):
        self.topic_change_threshold = topic_change_threshold
        self.threads: List[Dict[str, Any]] = []
        self.current_thread_id: Optional[str] = None
    
    def process_message(
        self,
        message: str,
        role: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Process a message and determine if it starts a new thread.
        
        Returns:
            Thread info with thread_id and is_new_thread flag
        """
        if not self.threads:
            # First message - create initial thread
            thread_id = self._generate_thread_id()
            self.threads.append({
                "thread_id": thread_id,
                "start_time": timestamp,
                "messages": [(role, message, timestamp)],
                "topics": self._extract_topics(message)
            })
            self.current_thread_id = thread_id
            
            return {
                "thread_id": thread_id,
                "is_new_thread": True,
                "reason": "initial"
            }
        
        # Check for topic change
        current_thread = self._get_thread(self.current_thread_id)
        topic_similarity = self._calculate_topic_similarity(
            current_thread["topics"],
            self._extract_topics(message)
        )
        
        is_new_thread = topic_similarity < self.topic_change_threshold
        
        if is_new_thread:
            # Start new thread
            thread_id = self._generate_thread_id()
            self.threads.append({
                "thread_id": thread_id,
                "start_time": timestamp,
                "messages": [(role, message, timestamp)],
                "topics": self._extract_topics(message)
            })
            self.current_thread_id = thread_id
            
            return {
                "thread_id": thread_id,
                "is_new_thread": True,
                "reason": "topic_change",
                "similarity": topic_similarity
            }
        else:
            # Continue current thread
            current_thread["messages"].append((role, message, timestamp))
            
            return {
                "thread_id": self.current_thread_id,
                "is_new_thread": False,
                "similarity": topic_similarity
            }
    
    def _generate_thread_id(self) -> str:
        """Generate unique thread ID."""
        return f"thread_{len(self.threads)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Get thread by ID."""
        for thread in self.threads:
            if thread["thread_id"] == thread_id:
                return thread
        return None
    
    def _extract_topics(self, text: str) -> Set[str]:
        """Extract key topics from text (simple keyword extraction)."""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Get top 5 most common
        counter = Counter(keywords)
        return set(word for word, _ in counter.most_common(5))
    
    def _calculate_topic_similarity(self, topics1: Set[str], topics2: Set[str]) -> float:
        """Calculate Jaccard similarity between topic sets."""
        if not topics1 or not topics2:
            return 0.0
        
        intersection = len(topics1 & topics2)
        union = len(topics1 | topics2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_thread_summary(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific thread."""
        thread = self._get_thread(thread_id)
        
        if not thread:
            return None
        
        return {
            "thread_id": thread_id,
            "start_time": thread["start_time"].isoformat(),
            "message_count": len(thread["messages"]),
            "topics": list(thread["topics"]),
            "duration": (thread["messages"][-1][2] - thread["start_time"]).seconds if len(thread["messages"]) > 1 else 0
        }


class MetadataEnricher:
    """
    Feature 6: Metadata Enrichment
    Add human-readable context to search results.
    """
    
    def enrich_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add enriched metadata to a search result."""
        metadata = result.get("metadata", {})
        
        enriched = {
            **result,
            "enriched_metadata": {
                "human_timestamp": self._format_timestamp(metadata.get("timestamp")),
                "age_description": self._describe_age(metadata.get("timestamp")),
                "quality_label": self._label_quality(metadata.get("quality_score", 0.7)),
                "context_summary": self._summarize_context(metadata),
                "relevance_explanation": self._explain_relevance(result.get("score", 0.5))
            }
        }
        
        return enriched
    
    def _format_timestamp(self, timestamp_str: Optional[str]) -> str:
        """Format timestamp for human reading."""
        if not timestamp_str:
            return "Unknown time"
        
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        except:
            return timestamp_str
    
    def _describe_age(self, timestamp_str: Optional[str]) -> str:
        """Describe how old the content is."""
        if not timestamp_str:
            return "Unknown age"
        
        try:
            dt = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - dt
            
            if age.days == 0:
                return "Today"
            elif age.days == 1:
                return "Yesterday"
            elif age.days < 7:
                return f"{age.days} days ago"
            elif age.days < 30:
                return f"{age.days // 7} weeks ago"
            elif age.days < 365:
                return f"{age.days // 30} months ago"
            else:
                return f"{age.days // 365} years ago"
        except:
            return "Unknown age"
    
    def _label_quality(self, quality_score: float) -> str:
        """Convert quality score to human label."""
        if quality_score >= 0.8:
            return "High quality"
        elif quality_score >= 0.6:
            return "Good quality"
        elif quality_score >= 0.4:
            return "Fair quality"
        else:
            return "Lower quality"
    
    def _summarize_context(self, metadata: Dict[str, Any]) -> str:
        """Create context summary from metadata."""
        parts = []
        
        if "mode" in metadata:
            parts.append(f"{metadata['mode']} mode")
        
        if "topics" in metadata:
            topics = metadata["topics"]
            if topics:
                parts.append(f"Topics: {', '.join(topics[:3])}")
        
        return " | ".join(parts) if parts else "General context"
    
    def _explain_relevance(self, score: float) -> str:
        """Explain relevance score."""
        if score >= 0.9:
            return "Highly relevant match"
        elif score >= 0.7:
            return "Very relevant"
        elif score >= 0.5:
            return "Moderately relevant"
        elif score >= 0.3:
            return "Somewhat relevant"
        else:
            return "Low relevance"


class QueryExpander:
    """
    Feature 7: Query Expansion
    Expand queries with synonyms and related terms.
    """
    
    def __init__(self):
        # Domain-specific synonym map
        self.synonym_map = {
            "doctor": ["physician", "medical professional", "clinician"],
            "patient": ["individual", "person", "client"],
            "medication": ["medicine", "drug", "prescription"],
            "error": ["bug", "issue", "problem", "exception"],
            "function": ["method", "procedure", "routine"],
            "fix": ["repair", "resolve", "correct", "patch"],
            "feeling": ["emotion", "sentiment", "mood"],
            "happy": ["joyful", "pleased", "content", "glad"],
            "sad": ["unhappy", "depressed", "down", "melancholy"]
        }
    
    def expand_query(
        self,
        query: str,
        max_expansions: int = 3
    ) -> Tuple[str, List[str]]:
        """
        Expand query with synonyms and related terms.
        
        Returns:
            (expanded_query, expansion_terms)
        """
        words = query.lower().split()
        expansions = []
        
        for word in words:
            if word in self.synonym_map:
                # Add top N synonyms
                synonyms = self.synonym_map[word][:max_expansions]
                expansions.extend(synonyms)
        
        # Create expanded query
        if expansions:
            expanded = query + " " + " ".join(expansions)
        else:
            expanded = query
        
        return expanded, expansions
    
    def add_synonyms(self, term: str, synonyms: List[str]):
        """Add custom synonyms to the map."""
        if term.lower() not in self.synonym_map:
            self.synonym_map[term.lower()] = []
        
        self.synonym_map[term.lower()].extend(synonyms)
    
    def get_expanded_terms(self, query: str) -> List[str]:
        """Get all expanded terms for a query."""
        _, expansions = self.expand_query(query)
        return expansions


if __name__ == "__main__":
    print("🌟 The Nexus Core - Enhancement Features Demo\n")
    
    # Demo 1: Citation Manager
    print("1️⃣ Citation Manager")
    cm = CitationManager()
    cm.add_citation("resp_001", "doc_123", "conversation", 0.92, "The patient shows symptoms...")
    cm.add_citation("resp_001", "doc_456", "medical_record", 0.87, "Previous diagnosis indicates...")
    print(cm.format_citations("resp_001", "numbered"))
    print(f"Quality: {cm.get_citation_quality('resp_001')}\n")
    
    # Demo 2: Context Window Manager
    print("2️⃣ Context Window Manager")
    cwm = ContextWindowManager(max_tokens=100)
    messages = [
        {"role": "user", "content": "Tell me about Python"},
        {"role": "assistant", "content": "Python is a high-level programming language..."},
        {"role": "user", "content": "What about JavaScript?"},
        {"role": "assistant", "content": "JavaScript is primarily used for web development..."}
    ]
    compressed = cwm.compress_context(messages)
    print(f"Compressed from {len(messages)} to {len(compressed)} messages\n")
    
    # Demo 3: Deduplication
    print("3️⃣ Deduplication Engine")
    de = DeduplicationEngine()
    results = [
        {"text": "The quick brown fox"},
        {"text": "The quick brown fox"},  # Exact duplicate
        {"text": "The quick brown fox jumps over"}  # Near duplicate
    ]
    deduped = de.deduplicate_results(results, method="hybrid")
    print(f"Deduplicated from {len(results)} to {len(deduped)} results\n")
    
    # Demo 4: Relevance Ranker
    print("4️⃣ Relevance Ranker")
    rr = RelevanceRanker()
    results = [
        {"text": "Old result", "score": 0.9, "metadata": {"timestamp": "2023-01-01T00:00:00"}},
        {"text": "Recent result", "score": 0.7, "metadata": {"timestamp": "2024-03-15T00:00:00", "quality_score": 0.95}}
    ]
    reranked = rr.rerank_results(results, "test query")
    for r in reranked:
        print(f"Score: {r['reranked_score']:.3f} - {r['text']}")
    print()
    
    # Demo 5: Thread Tracker
    print("5️⃣ Thread Tracker")
    tt = ConversationThreadTracker()
    result1 = tt.process_message("Let's talk about Python programming", "user", datetime.now())
    print(f"Message 1: Thread {result1['thread_id']}, New: {result1['is_new_thread']}")
    result2 = tt.process_message("What about error handling in Python?", "user", datetime.now())
    print(f"Message 2: Thread {result2['thread_id']}, New: {result2['is_new_thread']}")
    result3 = tt.process_message("I'm feeling stressed today", "user", datetime.now())
    print(f"Message 3: Thread {result3['thread_id']}, New: {result3['is_new_thread']}\n")
    
    # Demo 6: Metadata Enricher
    print("6️⃣ Metadata Enricher")
    me = MetadataEnricher()
    result = {
        "text": "Sample text",
        "score": 0.85,
        "metadata": {"timestamp": "2024-03-15T10:30:00", "quality_score": 0.9, "mode": "companion"}
    }
    enriched = me.enrich_result(result)
    print(f"Human timestamp: {enriched['enriched_metadata']['human_timestamp']}")
    print(f"Age: {enriched['enriched_metadata']['age_description']}")
    print(f"Quality: {enriched['enriched_metadata']['quality_label']}\n")
    
    # Demo 7: Query Expander
    print("7️⃣ Query Expander")
    qe = QueryExpander()
    expanded, terms = qe.expand_query("doctor medication error")
    print(f"Original: doctor medication error")
    print(f"Expanded: {expanded}")
    print(f"Expansion terms: {terms}")
