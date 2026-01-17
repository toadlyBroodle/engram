## Baseline: evaluate_gemini.sh: gemini-2.0.flash

Total number of questions and corresponding accuracy in each category: 
Single-hop:  4 841 560.3169999999998 0.666
Multi-hop:   1 282 106.71700000000001 0.378
Temporal:    2 321 135.20900000000006 0.421
Open-domain: 3 96 20.369 0.212
Adversarial:  5 446 350 0.785
Overall accuracy:  0.59

--------------------------------

# Baseline: gemini-2.0-flash-lite

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

--------------------------------

# Engram RLM (gemini-2.0-flash-lite) - Quick Test

3 QA samples evaluated; 3 accuracy values
Total number of questions and corresponding accuracy in each category: 
Temporal:    2 2 0.5 0.25
Open-domain: 3 1 0 0.0
Overall accuracy:  0.167

Note: Quick sanity test (1 conv, 3 QA). Model failed to use tools reliably.
Search finds answers but model claims "I don't have information".

══════════════════════════════════════════════════
📊 TOKEN USAGE STATS
══════════════════════════════════════════════════
🧠 Brain (gemini-2.0-flash-lite):
   Calls: 4
   Tokens: 2,804 in / 63 out = 2,867
   Cost: $0.0002
💾 MemMan (gemini-2.0-flash-lite):
   Calls: 113
   Tokens: 59,357 in / 36,111 out = 95,468
   Cost: $0.0153
──────────────────────────────────────────────────
📈 TOTAL:
   Calls: 117
   Tokens: 98,335
   Cost: $0.0155
══════════════════════════════════════════════════
--------------------------------

# Engram RLM v1 (gemini-2.0-flash + flash-lite) - 1 conv, 5 QA

5 QA samples evaluated; 733 memories ingested
Total number of questions and corresponding accuracy in each category: 
Multi-hop:   1 2 0.731 0.365
Temporal:    2 2 0.5 0.25
Open-domain: 3 1 0.105 0.105
Overall accuracy:  0.267

Analysis:
- Correct answers but TOO VERBOSE (full sentences vs short phrases)
- F1 score penalizes extra words heavily
- Search found wrong memory for "sunrise" (got "sunset" instead)

══════════════════════════════════════════════════
📊 TOKEN USAGE STATS
══════════════════════════════════════════════════
🧠 Brain (gemini-2.0-flash):
   Calls: 11
   Tokens: 9,967 in / 181 out = 10,148
   Cost: $0.0011
💾 MemMan (gemini-2.0-flash-lite):
   Calls: 214
   Tokens: 112,811 in / 72,265 out = 185,076
   Cost: $0.0301
──────────────────────────────────────────────────
📈 TOTAL:
   Calls: 225
   Tokens: 195,224
   Cost: $0.0312
══════════════════════════════════════════════════

Fixes applied for v2:
1. Better answer prompt - asks for 1-5 word SHORT answers
2. Improved search strategy in system prompt
3. Increased max_tool_calls from 5 to 8
