# Panel demo script (about 6 minutes)

**Before you start (2 minutes, every time):** run `python scripts/groq_check.py` (keys and models OK), start the server, open `/assistant` and `/dashboard` in two tabs. Delete nothing: a fresh wallet is ₹5,000 minus whatever is already in the log. Rehearse once with the same queries below, because live shops change.

1. **Ask.** "Wireless earbuds under ₹2000". Point out: pages come from search results and each page was opened and read; the score is shown; risk tag has reasons.
2. **Rules are hard, not hints.** "I want a non-Indian brand". Indian brands are dropped and the reply says what was left out and why; unknown brands are labelled "origin not verified".
3. **Allow path.** Buy a cheap item from a known shop. Gate says Allowed, receipt DEMO-xxxx, wallet drops. Say: simulated money, real links and prices.
4. **Escalate path.** Buy something over ₹1,500 (a cordless drill under ₹3,000 works). Gate says "Needs your approval" with the reason. Click Approve, then show nothing was charged before that click.
5. **Risk demo (best for showing every outcome).** Click the risk-demo chip on the start screen. Seven SYNTHETIC listings appear (named "[SYNTHETIC] …"). Press Buy on each: a normal one is Allowed; a look-alike shop, fake-looking reviews and a mid-price item go to approval; a dead link, a hidden "ignore previous instructions" in the listing and a ₹52,000 laptop are blocked. Say clearly that the data is invented while the rules are the real ones.
6. **Block path (real data).** Ask for a ₹20 face wash or paste a shady link such as a `.xyz` shop. Nothing settles and the reasons are on screen.
7. **Governance.** Open `/dashboard`: wallet, gate decision counts, the activity log, "chain intact". Optional proof: edit one amount in `data/audit/purchases.jsonl` and reload; the chain shows BROKEN at that entry. Restore the file afterwards.

## If something fails live
- *"Groq rate limit"*: wait a minute; the free tier is per minute. Do step 1 early to warm up.
- *Shop blocks reading (no price)*: say it out loud, this is the design: no verified price means Buy is blocked. Switch to another result.
- *No internet*: show the dashboard log and the tests (`python -m pytest tests -q`, all pass).

## Questions to expect
- **Is the payment real?** No. Demo wallet; the ledger is in the hash-chained audit log. Real payments would need a payment provider; `app/payments.py` already has a Razorpay test-mode path from the earlier pipeline.
- **Why not ASIDE here?** The web gate is a local rule set so the demo has no extra service to run; the ASIDE-backed pipeline is still in the repo.
- **How do you know a brand's country?** A reviewed lookup of about 100 brands; unknown ones are flagged, never guessed.
- **Can the model approve its own purchase?** No. Only the human's Approve click settles an escalated order; the price used is the server's own copy, not what the browser sends.
- **What is not covered?** Real payments, returns, shop accounts, reading shops that block bots, a complete brand database.
