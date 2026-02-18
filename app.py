from flask import Flask, request, jsonify
import hashlib
import time
import json
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
import os
import gunicorn

app = Flask(__name__)

# Cache class with LRU + TTL
class LRUCache:
    def __init__(self, max_size=2000, ttl_hours=24):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)

    def get(self, key):
        if key not in self.cache:
            return None
        entry = self.cache[key]
        if datetime.now() - entry['timestamp'] > self.ttl:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return entry['value']

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = {'value': value, 'timestamp': datetime.now()}
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def size(self):
        return len(self.cache)

# Global stores
exact_cache = LRUCache(max_size=2000, ttl_hours=24)
semantic_store = []
analytics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_tokens_would_have_used": 0,
    "tokens_saved": 0
}
AVG_TOKENS = 500
MODEL_COST_PER_M = 0.50

def normalize(query: str) -> str:
    return query.lower().strip()

def make_exact_key(query: str) -> str:
    normalized = normalize(query)
    return hashlib.md5(normalized.encode()).hexdigest()

def get_embedding(text: str):
    # Deterministic fake embedding for demo
    np.random.seed(abs(hash(text)) % (2**31))
    return np.random.rand(128).tolist()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_lookup(query_embedding, threshold=0.95):
    now = datetime.now()
    best_score, best_entry = 0, None
    for entry in semantic_store:
        if now - entry['timestamp'] > timedelta(hours=24):
            continue
        score = cosine_similarity(query_embedding, entry['embedding'])
        if score > best_score:
            best_score, best_entry = score, entry
    return best_entry['answer'] if best_score >= threshold else None

def semantic_store_add(query, embedding, answer):
    semantic_store.append({
        'query': query,
        'embedding': embedding,
        'answer': answer,
        'timestamp': datetime.now()
    })

def call_llm(query: str) -> str:
    time.sleep(0.5)  # Simulate LLM call
    return f"AI Answer: {query} (Generated at {datetime.now().strftime('%H:%M:%S')})"

@app.route("/", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query", "")
    
    analytics["total_requests"] += 1
    analytics["total_tokens_would_have_used"] += AVG_TOKENS
    
    start = time.time()
    cached_flag = False
    cache_key = make_exact_key(query_text)

    # Exact match first
    answer = exact_cache.get(cache_key)
    if answer:
        cached_flag = True
        analytics["cache_hits"] += 1
        analytics["tokens_saved"] += AVG_TOKENS
    else:
        # Semantic match
        embedding = get_embedding(normalize(query_text))
        answer = semantic_lookup(embedding)
        if answer:
            cached_flag = True
            analytics["cache_hits"] += 1
            analytics["tokens_saved"] += AVG_TOKENS
            exact_cache.set(cache_key, answer)
        else:
            # LLM call
            analytics["cache_misses"] += 1
            answer = call_llm(query_text)
            exact_cache.set(cache_key, answer)
            semantic_store_add(normalize(query_text), embedding, answer)

    latency_ms = int((time.time() - start) * 1000)

    return jsonify({
        "answer": answer,
        "cached": cached_flag,
        "latency": latency_ms,
        "cacheKey": cache_key
    })

@app.route("/analytics", methods=["GET"])
def analytics_endpoint():
    total = analytics["total_requests"]
    hits = analytics["cache_hits"]
    misses = analytics["cache_misses"]
    hit_rate = round(hits / total, 4) if total > 0 else 0.0
    
    cost_savings = round(analytics["tokens_saved"] * MODEL_COST_PER_M / 1_000_000, 2)
    baseline_cost = round(analytics["total_tokens_would_have_used"] * MODEL_COST_PER_M / 1_000_000, 2)
    savings_pct = round((cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0, 1)

    return jsonify({
        "hitRate": hit_rate,
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": exact_cache.size(),
        "costSavings": cost_savings,
        "savingsPercent": savings_pct,
        "strategies": ["exact match", "semantic similarity", "LRU eviction", "TTL expiration"]
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
