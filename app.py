from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib, time, json, os
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
CORS(app)

# All globals
exact_cache = OrderedDict()
semantic_store = []
analytics = {"total_requests":0,"cache_hits":0,"cache_misses":0,"total_tokens_would_have_used":0,"tokens_saved":0}
MAX_CACHE = 2000
TTL = timedelta(hours=24)
TOKENS = 500
COST_M = 0.50

def norm(q): return q.lower().strip()
def key(q): return hashlib.md5(norm(q).encode()).hexdigest()
def emb(q):
    np.random.seed(hash(q))
    return np.random.rand(128).tolist()
def sim(a,b):
    a,b=np.array(a),np.array(b)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def sem_lookup(e):
    now = datetime.now()
    for entry in semantic_store:
        if now-entry['ts'] > TTL: continue
        if sim(e, entry['emb']) > 0.95: return entry['ans']
    return None

def evict(): 
    global exact_cache
    while len(exact_cache) > MAX_CACHE: exact_cache.popitem(last=False)

@app.route("/analytics", methods=["GET"])
def ana():
    t,a,h,m = analytics["total_requests"],analytics["total_tokens_would_have_used"],analytics["tokens_saved"],analytics["cache_hits"]
    hr = round(h/t,4) if t else 0
    sv = round(analytics["tokens_saved"]*COST_M/1e6,2)
    bl = round(a*COST_M/1e6,2)
    sp = round(sv/bl*100,1) if bl else 0
    return jsonify({
        "hitRate":hr,"totalRequests":t,"cacheHits":analytics["cache_hits"],"cacheMisses":analytics["cache_misses"],
        "cacheSize":len(exact_cache),"costSavings":sv,"savingsPercent":sp,
        "strategies":["exact match","semantic similarity","LRU eviction","TTL expiration"]
    })

@app.route("/", methods=["GET", "POST"])  # ← GET + POST
def main():
    if request.method == "GET":
        return jsonify({"status":"OK","endpoints":["/ (POST queries)","/analytics (GET metrics)"]})
    
    data = request.get_json() or {}
    q = data.get("query", "")
    
    analytics["total_requests"] += 1
    analytics["total_tokens_would_have_used"] += TOKENS
    st = time.time()
    k = key(q)
    
    # EXACT
    ans = None
    if k in exact_cache:
        e = exact_cache[k]
        if datetime.now() - e['ts'] <= TTL:
            ans = e['ans']
            exact_cache.move_to_end(k)
    
    cached = ans is not None
    if not cached:
        # SEMANTIC
        e = emb(norm(q))
        ans = sem_lookup(e)
        if ans: cached = True
        
        if not cached:
            analytics["cache_misses"] += 1
            ans = f"AI: {q}"
            semantic_store.append({'emb':e,'ans':ans,'ts':datetime.now()})
    
    exact_cache[k] = {'ans':ans,'ts':datetime.now()}
    evict()
    
    if cached:
        analytics["cache_hits"] += 1
        analytics["tokens_saved"] += TOKENS
    
    return jsonify({
        "answer":ans,
        "cached":cached,
        "latency":int((time.time()-st)*1000),
        "cacheKey":k[:12]+"..."
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=False)
