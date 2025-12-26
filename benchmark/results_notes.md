# Baseline gemini-2.0-flash-lite

199 QA samples evaluated; 199 accuracy values
Total number of questions and corresponding accuracy in each category: 
Single-hop:  4 70 41.82100000000001 0.597
Multi-hop:   1 32 8.283999999999999 0.259
Temporal:    2 37 9.737000000000002 0.263
Open-domain: 3 13 0.7909999999999999 0.061
Adversarial:  5 47 36 0.766
Overall accuracy:  0.486

============================================================
✅ Benchmark complete! (BASELINE)
Results: benchmark/results_test_baseline.json
Stats: benchmark/results_test_baseline_stats.json
============================================================

══════════════════════════════════════════════════
📊 TOKEN USAGE STATS
══════════════════════════════════════════════════
🧠 Brain (gemini-2.0-flash-lite):
   Calls: 199
   Tokens: 3,629,281 in / 1,924 out = 3,631,205
   Cost: $0.2728
💾 MemMan (unknown):
   Calls: 0
   Tokens: 0 in / 0 out = 0
   Cost: $0.0000
──────────────────────────────────────────────────
📈 TOTAL:
   Calls: 199
   Tokens: 3,631,205
   Cost: $0.2728
   Rate: 16074.6 tokens/sec
══════════════════════════════════════════════════

# Engram gemini-2.0-flash-lite

199 QA samples evaluated; 199 accuracy values
Total number of questions and corresponding accuracy in each category: 
Single-hop:  4 70 13.822000000000005 0.197
Multi-hop:   1 32 4.076 0.127
Temporal:    2 37 6.886 0.186
Open-domain: 3 13 1.093 0.084
Adversarial:  5 47 0 0.0
Overall accuracy:  0.13

============================================================
✅ Benchmark complete! (ENGRAM)
Results: benchmark/results_test_engram.json
Stats: benchmark/results_test_engram_stats.json
============================================================

══════════════════════════════════════════════════
📊 TOKEN USAGE STATS
══════════════════════════════════════════════════
🧠 Brain (unknown):
   Calls: 0
   Tokens: 0 in / 0 out = 0
   Cost: $0.0000
💾 MemMan (gemini-2.0-flash-lite):
   Calls: 114
   Tokens: 59,904 in / 38,122 out = 98,026
   Cost: $0.0159
──────────────────────────────────────────────────
📈 TOTAL:
   Calls: 114
   Tokens: 98,026
   Cost: $0.0159
   Rate: 319.7 tokens/sec
══════════════════════════════════════════════════

## evaluate_gemini.sh: gemini-2.0.flash

199 QA samples evaluated; 199 accuracy values
105 QA samples evaluated; 105 accuracy values
193 QA samples evaluated; 193 accuracy values
260 QA samples evaluated; 260 accuracy values
242 QA samples evaluated; 242 accuracy values
158 QA samples evaluated; 158 accuracy values
190 QA samples evaluated; 190 accuracy values
239 QA samples evaluated; 239 accuracy values
196 QA samples evaluated; 196 accuracy values
204 QA samples evaluated; 204 accuracy values
Total number of questions and corresponding accuracy in each category: 
Single-hop:  4 841 560.3169999999998 0.666
Multi-hop:   1 282 106.71700000000001 0.378
Temporal:    2 321 135.20900000000006 0.421
Open-domain: 3 96 20.369 0.212
Adversarial:  5 446 350 0.785
Overall accuracy:  0.59