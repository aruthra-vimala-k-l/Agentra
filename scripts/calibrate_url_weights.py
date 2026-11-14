"""Data-derived points for the URL-level risk signals (weight-of-evidence scorecard method).

Data: labelled phishing (1) vs legitimate (0) URLs, 10,000 rows, 50/50 (shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques,
DataFiles/5.urldata.csv; features follow Mohammad, Thabtah & McCluskey's phishing-feature taxonomy).
Method: points = FACTOR * ln(LR), LR = P(feature | phishing) / P(feature | legitimate), Laplace-smoothed (+0.5).
FACTOR = PDO / ln 2 with PDO = 20 ("every 20 points doubles the odds of phishing"), the standard credit-scorecard convention.
Maps to Agentra signals:  Have_IP -> odd_url (raw numeric host),  TinyURL -> shortener.
LIMITS: legitimate URLs here are popular domains (selection bias), so lifts are probably overstated; only URL-structure signals
can be calibrated this way -- there is no public data with joint labels for seller, rating or price signals.
Run:  py scripts/calibrate_url_weights.py   (needs internet for the first run; caches next to this file)
"""
import json, math, pathlib, urllib.request
import pandas as pd

URL = "https://raw.githubusercontent.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/master/DataFiles/5.urldata.csv"
CACHE = pathlib.Path(__file__).with_name("urldata_phishing.csv")
PDO = 20
FACTOR = PDO / math.log(2)
MAP = {"Have_IP": "odd_url", "TinyURL": "shortener", "Have_At": "(not an Agentra signal yet)", "Prefix/Suffix": "(not used: hyphens are common in real shop names)"}


def main():
    if not CACHE.exists():
        urllib.request.urlretrieve(URL, CACHE)
    d = pd.read_csv(CACHE)
    ph, lg = d[d.Label == 1], d[d.Label == 0]
    out = {"n": len(d), "phishing": len(ph), "legitimate": len(lg), "pdo": PDO, "factor": round(FACTOR, 2), "signals": {}}
    for col, sig in MAP.items():
        a, b = int((ph[col] == 1).sum()), int((lg[col] == 1).sum())
        p1, p0 = (a + .5) / (len(ph) + 1), (b + .5) / (len(lg) + 1)
        lr = p1 / p0
        # 95% interval on ln(LR) (Katz log method)
        se = math.sqrt(1 / (a + .5) + 1 / (b + .5) - 1 / (len(ph) + 1) - 1 / (len(lg) + 1))
        lo, hi = math.log(lr) - 1.96 * se, math.log(lr) + 1.96 * se
        out["signals"][col] = {"agentra_signal": sig, "in_phishing": a, "in_legit": b, "likelihood_ratio": round(lr, 2),
                               "points": round(FACTOR * math.log(lr)), "points_95ci": [round(FACTOR * lo), round(FACTOR * hi)]}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
