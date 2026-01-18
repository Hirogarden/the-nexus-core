# 🚀 Quick Start Guide - The Nexus Core

Get up and running in 5 minutes!

## Step 1: Installation

### Option A: Zero Dependencies (Basic Mode)
```bash
# Just download the files - no installation needed!
# Works immediately with Python 3.8+
```

### Option B: Full Features (Recommended)
```bash
pip install llama-index langchain
```

## Step 2: Run the Demo

```bash
# Test the core engine
python nexus_core_engine.py

# Test hierarchical indexing
python nexus_core_indexing.py

# Test enhancement features
python nexus_core_enhancements.py

# Test data source manager
python data_source_manager.py
```

## Step 3: Your First Conversation

Create a file called `my_first_nexus.py`:

```python
from nexus_core_engine import NexusCoreEngine
from datetime import datetime

# Initialize
engine = NexusCoreEngine("./my_conversations")

# Log some conversations
engine.log_conversation_turn(
    session_id="first_session",
    user_message="Hello! What can The Nexus Core do?",
    assistant_response="The Nexus Core provides hierarchical conversation logging, intelligent search, and 7 advanced RAG features!",
    metadata={"mode": "getting_started"}
)

engine.log_conversation_turn(
    session_id="first_session",
    user_message="How does the hierarchical indexing work?",
    assistant_response="It uses 4 layers: Summary, Time-based, Topic-based, and Keyword indices for 20x faster search!",
    metadata={"mode": "getting_started"}
)

# Search your conversations
results = engine.semantic_search("hierarchical indexing", top_k=3)

print("🔍 Search Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.4f}")
    print(f"   {result['text'][:100]}...")
    print()

print(f"✅ Conversations saved to: my_conversations/")
print(f"📁 Check the folder - it's organized by Year/Month/Day!")
```

Run it:
```bash
python my_first_nexus.py
```

## Step 4: Explore the Structure

After running, check your `my_conversations/` folder:
```
my_conversations/
└── conversations/
    └── 2026/
        └── 01/
            └── 17/
                └── first_session.md
```

## Step 5: Try Advanced Features

```python
from nexus_core_enhancements import (
    CitationManager,
    RelevanceRanker,
    QueryExpander
)

# Track where information came from
citations = CitationManager()
citations.add_citation("response_1", "conversation_123", "chat", 0.95, "Key info here")
print(citations.format_citations("response_1", "numbered"))

# Expand your queries automatically
expander = QueryExpander()
expanded, terms = expander.expand_query("doctor visit")
print(f"Expanded: {expanded}")

# Re-rank results by multiple signals
ranker = RelevanceRanker()
reranked_results = ranker.rerank_results(results, "your query")
```

## What's Next?

1. **Customize**: Adjust parameters like `max_tokens` in ContextWindowManager
2. **Integrate**: Import into your existing project
3. **Scale**: The system handles 10,000+ conversations efficiently
4. **Monitor**: Check quality scores to track conversation quality

## Common Use Cases

### Clinical Documentation
```python
engine = NexusCoreEngine("./clinical_notes")
engine.log_conversation_turn(
    session_id="patient_123",
    user_message="Patient reports chest pain",
    assistant_response="Immediate assessment recommended...",
    metadata={"mode": "clinical", "priority": "high"}
)
```

### Code Assistant
```python
engine = NexusCoreEngine("./code_help")
engine.log_conversation_turn(
    session_id="debug_session",
    user_message="Why is my Python function failing?",
    assistant_response="The issue is in line 42...",
    metadata={"mode": "developer", "language": "python"}
)
```

### Personal Journal
```python
engine = NexusCoreEngine("./journal")
engine.log_conversation_turn(
    session_id=datetime.now().strftime("%Y%m%d"),
    user_message="How am I feeling today?",
    assistant_response="I'm doing well, had a productive day...",
    metadata={"mode": "companion"}
)
```

## Troubleshooting

**Q: "LlamaIndex not available" warning?**
A: That's okay! The system works without it. Install with `pip install llama-index` for enhanced features.

**Q: Search returns no results?**
A: Make sure you've logged some conversations first. The search needs data to find!

**Q: Where is my data stored?**
A: By default in `./nexus_data/`. Check the hierarchical folder structure.

**Q: Is my data encrypted?**
A: Data is stored as plain markdown by default. Add your own encryption layer if needed for sensitive data.

## Need Help?

- 📖 Read the full [README.md](README.md)
- 🐛 Found a bug? Open an issue on GitHub
- 💡 Want a feature? Submit a pull request!

---

**Ready to build something amazing?** The Nexus Core is at your service! 🚀
