from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import time
import json
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # FIX 1: CORS

exact_cache = OrderedDict()
semantic_store = []
analytics = {
    "total_requests": 0, "cache_hits": 0, "cache_misses": 0,
    "total_tokens_would_have_used": 0, "tokens_saved": 0
}
MAX_CACHE_SIZE = 2000
TTL_HOURS = 24
AVG_TOKENS = 500
MODEL_COST_PER_M = 0.50

def normalize(query):
    return query.lower().strip()

def make_exact_key(query):
    return hashlib.md5(normalize(query).encode()).hexdigest()

def get_embedding(text):
    np.random.seed(abs(hash(text)) % (2**31))
    return np.random.rand(128).tolist()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_lookup(query_embedding):
    now = datetime.now()
    best_score = 0
    best_answer = None
    for entry in semantic_store:
        if now - entry['timestamp'] > timedelta(hours=TTL_HOURS):
            continue
        score = cosine_similarity(query_embedding, entry['embedding'])
        if score > best_score and score > 0.95:
            best_score = score
            best_answer = entry['answer']
    return best_answer

def evict_lru():
    global exact_cache
    while len(exact_cache) > MAX_CACHE_SIZE:
        exact_cache.popitem(last=False)

@app.route("/analytics", methods=["GET"])  # FIX 2: Explicit GET
def analytics():
    total = analytics["total_requests"]
    hit_rate = round(analytics["cache_hits"] / total, 4) if total else 0
    
    cost_savings = round(analytics["tokens_saved"] * MODEL_COST_PER_M / 1000000, 2)
    baseline = round(analytics["total_tokens_would_have_used"] * MODEL_COST_PER_M / 1000000, 2)
    savings_pct = round(cost_savings / baseline * 100, 1) if baseline else 0
    
    return jsonify({
        "hitRate": hit_rate,
        "totalRequests": total,
        "cacheHits": analytics["cache_hits"],
        "cacheMisses": analytics["cache_misses"],
        "cacheSize": len(exact_cache),
        "costSavings": cost_savings,
        "savingsPercent": savings_pct,
        "strategies": ["exact match", "semantic similarity", "LRU eviction", "TTL expiration"]
    })

@app.route("/", methods=["POST", "GET"])  # FIX 3: Allow both POST+GET
def query():
    if request.method == "GET":
        return jsonify({"error": "Use POST with JSON body"})
    
    data = request.get_json() or {}
    query_text = data.get("query", "")
    
    analytics["total_requests"] += 1
    analytics["total_tokens_would_have_used"] += AVG_TOKENS
    
    start_time = time.time()
    cache_key = make_exact_key(query_text)
    
    # EXACT MATCH (Strategy 1)
    answer = None
    if cache_key in exact_cache:
        entry = exact_cache[cache_key]
        if datetime.now() - entry['timestamp'] <= timedelta(hours=TTL_HOURS):
            answer = entry['value']
            exact_cache.move_to_end(cache_key)
    
    cached = answer is not None
    if not cached:
        # SEMANTIC MATCH (Strategy 2)
        embedding = get_embedding(normalize(query_text))
        answer = semantic_lookup(embedding)
        if answer:
            cached = True
        
        if not cached:
            # MISS: Fake LLM
            analytics["cache_misses"] += 1
            answer = f"✅ AI Response: {query_text}"
            # Store both caches
            semantic_store.append({
                'embedding': embedding, 'answer': answer, 
                'timestamp': datetime.now()
            })
    
    # Cache exact match (Strategy 3: LRU)
    exact_cache[cache_key] = {'value': answer, 'timestamp': datetime.now()}
    evict_lru()
    
    if cached:
        analytics["cache_hits"] += 1
        analytics["tokens_saved"] += AVG_TOKENS
    
    latency = int((time.time() - start_time) * 1000)
    
    return jsonify({
        "answer": answer,
        "cached": cached,
        "latency": latency,
        "cacheKey": cache_key[:16] + "..."
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
