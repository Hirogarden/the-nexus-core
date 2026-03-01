"""
The Nexus Core - HiRAG (Hierarchical RAG) Memory

4-layer compression pipeline:
  1. Ephemeral  — raw conversation turns      nexus_data/hirag/ephemeral.jsonl
  2. Daily      — per-day compressed summaries nexus_data/hirag/daily.jsonl
  3. Topic      — cross-day topic clusters     nexus_data/hirag/topics.jsonl
  4. Identity   — long-horizon user patterns   nexus_data/hirag/identity.jsonl

As memories age they are compressed upward through the layers.  Each layer
trades detail for breadth, giving the system both recent precision and
long-horizon continuity.

Zero external dependencies — stdlib only.
Compression uses extractive summarisation by default; pass a summarize_fn
callable for LLM-augmented compression.
"""

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from nexus_core_config import config as _config


# ---------------------------------------------------------------------------
# Layer 1 — Ephemeral turn
# ---------------------------------------------------------------------------

@dataclass
class EphemeralTurn:
    """A single raw conversation turn stored verbatim."""
    turn_id:    str
    session_id: str
    query:      str
    response:   str
    timestamp:  str          # ISO 8601 UTC
    compressed: bool = False # True once folded into a DailySummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id":    self.turn_id,
            "session_id": self.session_id,
            "query":      self.query,
            "response":   self.response,
            "timestamp":  self.timestamp,
            "compressed": self.compressed,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EphemeralTurn":
        return EphemeralTurn(
            turn_id=d["turn_id"],
            session_id=d.get("session_id", ""),
            query=d["query"],
            response=d["response"],
            timestamp=d["timestamp"],
            compressed=d.get("compressed", False),
        )


# ---------------------------------------------------------------------------
# Layer 2 — Daily summary
# ---------------------------------------------------------------------------

@dataclass
class DailySummary:
    """Compressed summary of all turns for one calendar day (UTC)."""
    summary_id:           str
    date:                 str        # YYYY-MM-DD
    summary_text:         str
    turn_count:           int
    key_topics:           List[str]
    compressed_at:        str        # ISO 8601 UTC
    source_turn_ids:      List[str]
    compressed_to_topic:  bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_id":          self.summary_id,
            "date":                self.date,
            "summary_text":        self.summary_text,
            "turn_count":          self.turn_count,
            "key_topics":          self.key_topics,
            "compressed_at":       self.compressed_at,
            "source_turn_ids":     self.source_turn_ids,
            "compressed_to_topic": self.compressed_to_topic,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DailySummary":
        return DailySummary(
            summary_id=d["summary_id"],
            date=d["date"],
            summary_text=d["summary_text"],
            turn_count=d["turn_count"],
            key_topics=d.get("key_topics", []),
            compressed_at=d["compressed_at"],
            source_turn_ids=d.get("source_turn_ids", []),
            compressed_to_topic=d.get("compressed_to_topic", False),
        )


# ---------------------------------------------------------------------------
# Layer 3 — Topic cluster
# ---------------------------------------------------------------------------

@dataclass
class TopicCluster:
    """A recurring topic or theme extracted from multiple daily summaries."""
    topic_id:                  str
    topic_name:                str
    description:               str
    related_dates:             List[str]
    confidence:                float     # 0.0–1.0
    created_at:                str
    compressed_to_identity:    bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id":               self.topic_id,
            "topic_name":             self.topic_name,
            "description":            self.description,
            "related_dates":          self.related_dates,
            "confidence":             self.confidence,
            "created_at":             self.created_at,
            "compressed_to_identity": self.compressed_to_identity,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TopicCluster":
        return TopicCluster(
            topic_id=d["topic_id"],
            topic_name=d["topic_name"],
            description=d["description"],
            related_dates=d.get("related_dates", []),
            confidence=d.get("confidence", 0.5),
            created_at=d["created_at"],
            compressed_to_identity=d.get("compressed_to_identity", False),
        )


# ---------------------------------------------------------------------------
# Layer 4 — Identity pattern
# ---------------------------------------------------------------------------

@dataclass
class IdentityPattern:
    """
    A stable, long-horizon pattern about the user distilled from topics.
    pattern_type: "recurring_interest" | "preference" |
                  "communication_style" | "knowledge_domain"
    """
    pattern_id:   str
    pattern_type: str
    description:  str
    evidence:     List[str]   # topic_ids
    strength:     float       # 0.0–1.0; increases with more evidence
    created_at:   str
    updated_at:   str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id":   self.pattern_id,
            "pattern_type": self.pattern_type,
            "description":  self.description,
            "evidence":     self.evidence,
            "strength":     self.strength,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IdentityPattern":
        return IdentityPattern(
            pattern_id=d["pattern_id"],
            pattern_type=d["pattern_type"],
            description=d["description"],
            evidence=d.get("evidence", []),
            strength=d.get("strength", 0.5),
            created_at=d["created_at"],
            updated_at=d.get("updated_at", d["created_at"]),
        )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "what", "how", "when",
    "where", "why", "which", "who", "not", "so", "if", "then", "also",
})


def _new_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:8]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_of(iso_ts: str) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp string."""
    return iso_ts[:10]


def _top_keywords(texts: List[str], n: int = 8) -> List[str]:
    """Return the top-N content words across a list of text strings."""
    words: List[str] = []
    for text in texts:
        words.extend(
            w.lower()
            for w in re.findall(r"[a-zA-Z]{4,}", text)
            if w.lower() not in _STOP_WORDS
        )
    return [w for w, _ in Counter(words).most_common(n)]


def _extractive_summary(turns: List[EphemeralTurn], max_len: int = 800) -> str:
    """Build a simple extractive summary from a list of turns."""
    parts = []
    for t in turns:
        q = t.query[:120].replace("\n", " ")
        r = t.response[:240].replace("\n", " ")
        parts.append(f"Q: {q}\nA: {r}")
    return "\n\n".join(parts)[:max_len]


def _keyword_score(query: str, text: str) -> float:
    """Simple keyword overlap score in [0.0, 1.0]."""
    q_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z]{3,}", query)
        if w.lower() not in _STOP_WORDS
    }
    t_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z]{3,}", text)
        if w.lower() not in _STOP_WORDS
    }
    if not q_words or not t_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, factory) -> list:
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(factory(json.loads(line)))
            except Exception:
                pass
    return items


def _append_jsonl(path: Path, obj) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj.to_dict()) + "\n")


def _rewrite_jsonl(path: Path, items: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# HiRAGMemory
# ---------------------------------------------------------------------------

class HiRAGMemory:
    """
    4-layer hierarchical memory with automatic compression.

    Parameters
    ----------
    data_dir
        Root nexus_data directory.  Uses config default if None.
    summarize_fn
        Optional ``callable(prompt: str) -> str`` for LLM-augmented
        compression.  Falls back to extractive summarisation when None.
    max_ephemeral
        Compress ephemeral → daily when uncompressed turn count reaches
        this threshold (default 50).
    topic_threshold
        Compress daily → topics when uncompressed daily summary count
        reaches this threshold (default 5).
    identity_threshold
        Compress topics → identity when uncompressed topic count reaches
        this threshold (default 3).
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        summarize_fn: Optional[Callable[[str], str]] = None,
        max_ephemeral: int = 50,
        topic_threshold: int = 5,
        identity_threshold: int = 3,
    ):
        base = Path(data_dir) if data_dir else Path(_config.nexus_data_path)
        self._dir = base / "hirag"
        self._dir.mkdir(parents=True, exist_ok=True)

        self._eph_path      = self._dir / "ephemeral.jsonl"
        self._daily_path    = self._dir / "daily.jsonl"
        self._topics_path   = self._dir / "topics.jsonl"
        self._identity_path = self._dir / "identity.jsonl"

        self._summarize_fn        = summarize_fn
        self._max_ephemeral       = max_ephemeral
        self._topic_threshold     = topic_threshold
        self._identity_threshold  = identity_threshold

        print(
            f"[hirag] Initialised — "
            f"eph:{max_ephemeral} daily:{topic_threshold} topic:{identity_threshold}"
        )

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_ephemeral(self) -> List[EphemeralTurn]:
        return _load_jsonl(self._eph_path, EphemeralTurn.from_dict)

    def _load_daily(self) -> List[DailySummary]:
        return _load_jsonl(self._daily_path, DailySummary.from_dict)

    def _load_topics(self) -> List[TopicCluster]:
        return _load_jsonl(self._topics_path, TopicCluster.from_dict)

    def _load_identity(self) -> List[IdentityPattern]:
        return _load_jsonl(self._identity_path, IdentityPattern.from_dict)

    # ------------------------------------------------------------------
    # Ingest  (Layer 1)
    # ------------------------------------------------------------------

    def ingest_turn(
        self,
        query: str,
        response: str,
        session_id: str = "",
    ) -> EphemeralTurn:
        """
        Append a conversation turn to the ephemeral layer.

        Does NOT trigger compression — call ``maybe_compress()`` after the
        query completes to avoid adding latency.
        """
        turn = EphemeralTurn(
            turn_id=_new_id(),
            session_id=session_id,
            query=query,
            response=response,
            timestamp=_now_iso(),
        )
        _append_jsonl(self._eph_path, turn)
        return turn

    # ------------------------------------------------------------------
    # Compression  Layer 1 → 2  (Ephemeral → Daily)
    # ------------------------------------------------------------------

    def compress_ephemeral_to_daily(self) -> List[DailySummary]:
        """
        Compress all uncompressed ephemeral turns into daily summaries.
        Turns are grouped by UTC calendar date.

        Returns the newly created DailySummary objects.
        """
        turns = self._load_ephemeral()
        uncompressed = [t for t in turns if not t.compressed]
        if not uncompressed:
            return []

        # Group by date
        by_date: Dict[str, List[EphemeralTurn]] = {}
        for t in uncompressed:
            by_date.setdefault(_date_of(t.timestamp), []).append(t)

        new_summaries: List[DailySummary] = []
        for date, day_turns in sorted(by_date.items()):
            summary_text = self._summarise_turns(day_turns)
            key_topics = _top_keywords(
                [t.query + " " + t.response for t in day_turns]
            )
            ds = DailySummary(
                summary_id=_new_id(),
                date=date,
                summary_text=summary_text,
                turn_count=len(day_turns),
                key_topics=key_topics,
                compressed_at=_now_iso(),
                source_turn_ids=[t.turn_id for t in day_turns],
            )
            _append_jsonl(self._daily_path, ds)
            new_summaries.append(ds)

        # Mark turns as compressed and rewrite
        compressed_dates = set(by_date.keys())
        for t in turns:
            if not t.compressed and _date_of(t.timestamp) in compressed_dates:
                t.compressed = True
        _rewrite_jsonl(self._eph_path, turns)

        print(
            f"[hirag] {len(uncompressed)} turns -> "
            f"{len(new_summaries)} daily summaries"
        )
        return new_summaries

    def _summarise_turns(self, turns: List[EphemeralTurn]) -> str:
        if self._summarize_fn is None:
            return _extractive_summary(turns)
        prompt = (
            "Summarise the following conversation turns into one concise paragraph "
            "capturing the main topics and key information exchanged.\n\n"
            + _extractive_summary(turns, max_len=1500)
            + "\n\nSummary:"
        )
        try:
            return self._summarize_fn(prompt).strip()
        except Exception:
            return _extractive_summary(turns)

    # ------------------------------------------------------------------
    # Compression  Layer 2 → 3  (Daily → Topics)
    # ------------------------------------------------------------------

    def compress_daily_to_topics(self) -> List[TopicCluster]:
        """
        Extract topic clusters from uncompressed daily summaries.
        Returns the newly created TopicCluster objects.
        """
        daily = self._load_daily()
        uncompressed = [d for d in daily if not d.compressed_to_topic]
        if not uncompressed:
            return []

        all_keywords: List[str] = []
        for ds in uncompressed:
            all_keywords.extend(ds.key_topics)

        if self._summarize_fn:
            topics = self._extract_topics_llm(uncompressed)
        else:
            topics = self._extract_topics_extractive(uncompressed, all_keywords)

        for t in topics:
            _append_jsonl(self._topics_path, t)

        ids = {d.summary_id for d in uncompressed}
        for d in daily:
            if d.summary_id in ids:
                d.compressed_to_topic = True
        _rewrite_jsonl(self._daily_path, daily)

        print(
            f"[hirag] {len(uncompressed)} daily summaries -> "
            f"{len(topics)} topic clusters"
        )
        return topics

    def _extract_topics_extractive(
        self, summaries: List[DailySummary], all_keywords: List[str]
    ) -> List[TopicCluster]:
        freq = Counter(all_keywords)
        top = [w for w, _ in freq.most_common(5)]
        if not top:
            return []
        dates = sorted({d.date for d in summaries})
        return [
            TopicCluster(
                topic_id=_new_id(),
                topic_name=kw,
                description=(
                    f"Recurring subject '{kw}' appeared in "
                    f"{freq[kw]} daily entr{'y' if freq[kw]==1 else 'ies'}."
                ),
                related_dates=dates,
                confidence=min(1.0, freq[kw] / len(summaries)),
                created_at=_now_iso(),
            )
            for kw in top
        ]

    def _extract_topics_llm(self, summaries: List[DailySummary]) -> List[TopicCluster]:
        text = "\n\n".join(
            f"[{d.date}] {d.summary_text[:300]}" for d in summaries
        )
        prompt = (
            "From the following daily conversation summaries, identify up to 5 "
            "distinct recurring topics or themes.  For each topic provide a short "
            "name and a one-sentence description.\n\n"
            f"{text}\n\n"
            "Respond as a numbered list:\n1. TopicName: description\n2. ..."
        )
        try:
            result = self._summarize_fn(prompt).strip()
            topics: List[TopicCluster] = []
            dates = sorted({d.date for d in summaries})
            for line in result.splitlines():
                m = re.match(r"\d+\.\s*(.+?):\s*(.+)", line)
                if m:
                    topics.append(TopicCluster(
                        topic_id=_new_id(),
                        topic_name=m.group(1).strip(),
                        description=m.group(2).strip(),
                        related_dates=dates,
                        confidence=0.7,
                        created_at=_now_iso(),
                    ))
            return topics if topics else self._extract_topics_extractive(
                summaries, [kw for d in summaries for kw in d.key_topics]
            )
        except Exception:
            return self._extract_topics_extractive(
                summaries, [kw for d in summaries for kw in d.key_topics]
            )

    # ------------------------------------------------------------------
    # Compression  Layer 3 → 4  (Topics → Identity)
    # ------------------------------------------------------------------

    def compress_topics_to_identity(self) -> List[IdentityPattern]:
        """
        Distil identity patterns from uncompressed topic clusters.
        Returns the newly created IdentityPattern objects.
        """
        topics = self._load_topics()
        uncompressed = [t for t in topics if not t.compressed_to_identity]
        if not uncompressed:
            return []

        if self._summarize_fn:
            patterns = self._extract_identity_llm(uncompressed)
        else:
            patterns = self._extract_identity_extractive(uncompressed)

        for p in patterns:
            _append_jsonl(self._identity_path, p)

        ids = {t.topic_id for t in uncompressed}
        for t in topics:
            if t.topic_id in ids:
                t.compressed_to_identity = True
        _rewrite_jsonl(self._topics_path, topics)

        print(
            f"[hirag] {len(uncompressed)} topics -> "
            f"{len(patterns)} identity patterns"
        )
        return patterns

    def _extract_identity_extractive(
        self, topics: List[TopicCluster]
    ) -> List[IdentityPattern]:
        high = [t for t in topics if t.confidence >= 0.5] or topics
        desc = ", ".join(t.topic_name for t in high[:5])
        return [IdentityPattern(
            pattern_id=_new_id(),
            pattern_type="recurring_interest",
            description=f"Frequently engages with: {desc}.",
            evidence=[t.topic_id for t in high],
            strength=min(1.0, len(high) / max(len(topics), 1)),
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )]

    def _extract_identity_llm(
        self, topics: List[TopicCluster]
    ) -> List[IdentityPattern]:
        text = "\n".join(f"- {t.topic_name}: {t.description}" for t in topics)
        prompt = (
            "From these recurring topics, identify up to 3 stable user patterns "
            "(recurring_interest / preference / communication_style / knowledge_domain). "
            "For each supply a type and one-sentence description.\n\n"
            f"{text}\n\n"
            "Respond as a numbered list:\n1. pattern_type: description\n2. ..."
        )
        try:
            result = self._summarize_fn(prompt).strip()
            patterns: List[IdentityPattern] = []
            for line in result.splitlines():
                m = re.match(r"\d+\.\s*(.+?):\s*(.+)", line)
                if m:
                    patterns.append(IdentityPattern(
                        pattern_id=_new_id(),
                        pattern_type=m.group(1).strip().lower().replace(" ", "_"),
                        description=m.group(2).strip(),
                        evidence=[t.topic_id for t in topics],
                        strength=0.6,
                        created_at=_now_iso(),
                        updated_at=_now_iso(),
                    ))
            return patterns if patterns else self._extract_identity_extractive(topics)
        except Exception:
            return self._extract_identity_extractive(topics)

    # ------------------------------------------------------------------
    # Auto-compression trigger
    # ------------------------------------------------------------------

    def maybe_compress(self) -> Dict[str, int]:
        """
        Check each compression threshold and run whichever passes.

        Returns a dict with counts of items compressed at each stage:
        ``{"daily": N, "topics": N, "identity": N}``

        Call this *after* responding to a query so it never adds latency.
        """
        result = {"daily": 0, "topics": 0, "identity": 0}

        eph = self._load_ephemeral()
        if sum(1 for t in eph if not t.compressed) >= self._max_ephemeral:
            result["daily"] = len(self.compress_ephemeral_to_daily())

        daily = self._load_daily()
        if sum(1 for d in daily if not d.compressed_to_topic) >= self._topic_threshold:
            result["topics"] = len(self.compress_daily_to_topics())

        topics = self._load_topics()
        if sum(1 for t in topics if not t.compressed_to_identity) >= self._identity_threshold:
            result["identity"] = len(self.compress_topics_to_identity())

        return result

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search across all 4 layers and return ranked context snippets.

        Each result dict contains:
          layer    — "ephemeral" | "daily" | "topic" | "identity"
          content  — text ready to prepend to the LLM prompt
          score    — relevance in [0.0, 1.0]
          metadata — layer-specific fields
        """
        results: List[Dict[str, Any]] = []

        # Layer 1: last 10 uncompressed turns — always include with recency bonus
        eph = self._load_ephemeral()
        recent = [t for t in eph if not t.compressed][-10:]
        for t in recent:
            score = _keyword_score(query, t.query + " " + t.response)
            score = max(score, 0.1)   # recency baseline
            results.append({
                "layer":   "ephemeral",
                "content": (
                    f"[{_date_of(t.timestamp)}] "
                    f"Q: {t.query[:150]} | A: {t.response[:300]}"
                ),
                "score":    round(score, 3),
                "metadata": {"turn_id": t.turn_id, "timestamp": t.timestamp},
            })

        # Layer 2: daily summaries — keyword match
        for ds in self._load_daily():
            score = _keyword_score(
                query, ds.summary_text + " " + " ".join(ds.key_topics)
            )
            if score > 0:
                results.append({
                    "layer":   "daily",
                    "content": f"[Daily {ds.date}] {ds.summary_text[:400]}",
                    "score":    round(score * 0.85, 3),
                    "metadata": {"summary_id": ds.summary_id, "date": ds.date},
                })

        # Layer 3: topic clusters — keyword match weighted by confidence
        for tc in self._load_topics():
            score = _keyword_score(query, tc.topic_name + " " + tc.description)
            if score > 0:
                results.append({
                    "layer":   "topic",
                    "content": f"[Topic: {tc.topic_name}] {tc.description}",
                    "score":    round(score * tc.confidence * 0.7, 3),
                    "metadata": {
                        "topic_id":   tc.topic_id,
                        "topic_name": tc.topic_name,
                    },
                })

        # Layer 4: identity — always included at low base weight
        for ip in self._load_identity():
            score = _keyword_score(query, ip.description)
            base  = 0.15 * ip.strength
            results.append({
                "layer":   "identity",
                "content": f"[{ip.pattern_type}] {ip.description}",
                "score":    round(max(score * 0.5, base), 3),
                "metadata": {
                    "pattern_id":   ip.pattern_id,
                    "pattern_type": ip.pattern_type,
                },
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        eph      = self._load_ephemeral()
        daily    = self._load_daily()
        topics   = self._load_topics()
        identity = self._load_identity()

        unc_eph    = sum(1 for t in eph if not t.compressed)
        unc_daily  = sum(1 for d in daily if not d.compressed_to_topic)
        unc_topics = sum(1 for t in topics if not t.compressed_to_identity)

        return {
            "ephemeral": {
                "total_turns":        len(eph),
                "uncompressed_turns": unc_eph,
            },
            "daily": {
                "total_summaries":       len(daily),
                "uncompressed_to_topic": unc_daily,
            },
            "topic": {
                "total_clusters":            len(topics),
                "uncompressed_to_identity":  unc_topics,
            },
            "identity": {
                "total_patterns": len(identity),
            },
            "compression_pending": {
                "ephemeral_to_daily":   unc_eph    >= self._max_ephemeral,
                "daily_to_topics":      unc_daily  >= self._topic_threshold,
                "topics_to_identity":   unc_topics >= self._identity_threshold,
            },
        }
