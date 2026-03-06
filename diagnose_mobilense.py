import json, sys
sys.path.insert(0, '.')
from pathlib import Path
from nexus_core_ingestion import (
    _keyword_score, _keyword_candidate_scores,
    _get_embedding_cache, _EMBEDDING_MODEL, _np,
    _HYBRID_VECTOR_WEIGHT, _HYBRID_KEYWORD_WEIGHT,
    ChunkStore, search_knowledge_base
)
from nexus_core_config import config

query = 'What is the hardiness zone of Allium canadense mobilense?'
kb_dir = Path(config.nexus_data_path) / 'knowledge_base'
store = ChunkStore(kb_dir)

# Find mobilense chunk
mobilense_chunk = None
with store.chunks_file.open(encoding='utf-8') as fh:
    for line in fh:
        if 'mobilense' in line:
            c = json.loads(line.strip())
            if 'Allium canadense mobilense' in c.get('text', ''):
                mobilense_chunk = c
                break

print(f'chunk_id: {mobilense_chunk["chunk_id"]}')
ks = _keyword_score(query, mobilense_chunk['text'])

matrix, cache_ids = _get_embedding_cache(store)
cache_id_to_idx = {cid: i for i, cid in enumerate(cache_ids)}
q_emb = _EMBEDDING_MODEL.encode([query], normalize_embeddings=True, batch_size=1)
q_vec = _np.array(q_emb[0], dtype='float32')
scores = matrix @ q_vec
idx = cache_id_to_idx.get(mobilense_chunk['chunk_id'], -1)
vs = float(scores[idx]) if idx >= 0 else -1
hybrid = _HYBRID_VECTOR_WEIGHT * vs + _HYBRID_KEYWORD_WEIGHT * ks
print(f'vector_score: {vs:.4f}  keyword_score: {ks:.4f}  hybrid: {hybrid:.4f}')

print('\n=== Top 5 search results ===')
results = search_knowledge_base(query, top_k=5)
for i, r in enumerate(results):
    name = r['text'][:60].replace('\n', ' ')
    print(f'  {i+1}. score={r["score"]}  {name}')
