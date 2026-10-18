# Benchmarks (0-shot, lm-eval 0.4.13)

acc and acc_norm in %; answer BPB = bits per UTF-8 byte of the gold answer (lower is better).
Runs with a `limit` are partial: none.

| task | metric | random | fw60_py20_fm13_gb07_s1234 | gpt2 | random_init |
|---|---|---|---|---|---|
| hellaswag | acc |  25.0 | 26.0 | 28.9 | 25.6 |
| hellaswag | acc_norm |  25.0 | 26.0 | 31.1 | 26.3 |
| hellaswag | answer BPB |  | 1.895 | 0.964 | 3.410 |
| piqa | acc |  50.0 | 51.5 | 62.9 | 53.5 |
| piqa | acc_norm |  50.0 | 50.4 | 62.5 | 52.1 |
| piqa | answer BPB |  | 2.201 | 1.123 | 3.585 |
| arc_easy | acc |  25.0 | 27.2 | 43.8 | 24.4 |
| arc_easy | acc_norm |  25.0 | 28.0 | 39.5 | 25.6 |
| arc_easy | answer BPB |  | 1.912 | 1.248 | 2.884 |
| sciq | acc |  25.0 | 22.3 | 75.2 | 19.8 |
| sciq | acc_norm |  25.0 | 25.2 | 64.4 | 23.1 |
| sciq | answer BPB |  | 2.322 | 1.405 | 2.501 |
| lambada_openai | acc |   0.0 | 0.0 | 32.6 | 0.0 |
| lambada_openai | acc_norm |   0.0 | - | - | - |
| lambada_openai | answer BPB |  | 3.075 | 0.797 | 3.215 |
