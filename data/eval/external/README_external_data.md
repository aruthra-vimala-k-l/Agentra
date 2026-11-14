# External evaluation data (not written by Agentra)
- `phishtank_verified_urls.txt`: 14,858 verified, online phishing URLs from PhishTank (submitted around May 2020). PhishTank data is free to use with attribution.
- `benign_urls.txt`: 35,378 URLs from a benign-site list (mostly older http URLs; only ~5.5% https), as distributed in
  shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques (DataFiles/1.Benign_list_big_final.csv, 2.online-valid.csv).
Used only by scripts/evaluate_risk_detectors.py. Known limits: the benign list is dated and is not shopping URLs; its low https share
makes "insecure link" look useless on it (a dataset age artefact), so the evaluation also reports results without that signal.
