import sys, random, json, statistics as st
sys.path.insert(0,'.')
from app.web_assistant import RISK_POINTS as W, HIGH_RISK_AT as T
# plausible firing rates for a listing on a trusted marketplace (most rules rarely fire)
P={k:0.05 for k in W}
P.update(seller_unknown=.35, seller_unproven=.25, no_rating_evidence=.12, thin_ratings=.15, no_price=.05, no_rating=.1,
         low_rating=.1, over_budget=.2, too_good_few=.06, price_too_low=.05, extreme_discount=.08, prompt_injection=.01,
         lookalike_domain=.01, insecure_url=.01, odd_url=.01, shortener=.01, risky_tld=.02, dead_link=.02, unverified_source=.1,
         review_flagged=.05, not_product_page=.05, wrong_region=.03)
rng=random.Random(7)
def verdict(fired,w): return min(100,sum(w[k] for k in fired))>=T
N=20000; res={}
for pct in (0.1,0.2,0.3):
    flips=0; hi=0
    for _ in range(N):
        fired=[k for k in W if rng.random()<P[k]]
        base=verdict(fired,W); hi+=base
        w={k:v*(1+rng.uniform(-pct,pct)) for k,v in W.items()}
        flips+= (verdict(fired,w)!=base)
    res[f"+/-{int(pct*100)}% every weight"]=dict(flip_rate=round(flips/N,4), baseline_high_share=round(hi/N,3))
# threshold sensitivity
thr={}
for t in (30,35,40,45,50):
    c=0
    for _ in range(N):
        fired=[k for k in W if rng.random()<P[k]]
        c+= min(100,sum(W[k] for k in fired))>=t
    thr[t]=round(c/N,3)
# single-signal check: which signals alone cross the threshold, and which pairs of mid signals do
alone=[k for k,v in W.items() if v>=T]
mid=[k for k,v in W.items() if 15<=v<T]
pairs=sum(1 for i,a in enumerate(mid) for b in mid[i+1:] if W[a]+W[b]>=T)
print(json.dumps(dict(threshold=T, signals_that_alone_reach_high=alone, mid_signals=len(mid), mid_pairs_reaching_high=pairs, sensitivity=res, high_share_by_threshold=thr),indent=1))
