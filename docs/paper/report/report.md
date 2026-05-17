# PROPHET — comparison report

## Headline (95% CI)

| agent | n | acc | ECE | Brier | log-loss | net payoff |
|---|---|---|---|---|---|---|
| baseline:random | 153 | 0.000 [0.000,0.000] | 0.532 [0.485,0.579] | 0.367 [0.317,0.417] | 1.140 [0.956,1.330] | -11604.2 [-13117.6,-10189.5] |
| baseline:always-take | 240 | 0.000 [0.000,0.000] | 0.990 [0.990,0.990] | 0.980 [0.980,0.980] | 4.605 [4.605,4.605] | -24369.8 [-25428.9,-23395.0] |
| baseline:oracle | 240 | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 38684.7 [38031.9,39342.8] |
| openrouter:openai_gpt-5.4-nano | 229 | 0.734 [0.677,0.786] | 0.132 [0.097,0.193] | 0.191 [0.153,0.233] | 0.656 [0.516,0.812] | 11366.2 [8738.0,13888.6] |
| openrouter:google_gemini-3.1-flash-lite | 230 | 0.835 [0.787,0.883] | 0.165 [0.117,0.213] | 0.165 [0.117,0.213] | 3.424 [2.433,4.415] | 15310.7 [12910.0,17588.5] |
| openrouter:google_gemini-3-flash-preview | 231 | 0.970 [0.944,0.991] | 0.027 [0.008,0.051] | 0.027 [0.009,0.051] | 0.542 [0.181,0.997] | 22907.7 [21527.7,24106.9] |
| openrouter:anthropic_claude-haiku-4.5 | 230 | 0.935 [0.900,0.965] | 0.044 [0.018,0.080] | 0.061 [0.034,0.092] | 0.267 [0.154,0.401] | 21739.2 [19913.9,23421.8] |
| openrouter:meta-llama_llama-4-scout | 207 | 0.831 [0.778,0.879] | 0.147 [0.105,0.203] | 0.158 [0.112,0.208] | 2.717 [1.823,3.669] | 12242.5 [9783.6,14494.5] |

## Pairwise significance (Holm-Bonferroni adjusted)

| A | B | metric | p_raw | reject H0? |
|---|---|---|---|---|
| baseline:random | baseline:always-take | payoff_t | 0.0000 | **yes** |
| baseline:random | baseline:always-take | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | baseline:always-take | accuracy_mcnemar | 1.0000 | no |
| baseline:random | baseline:always-take | ece_permutation | 0.0005 | **yes** |
| baseline:random | baseline:always-pass | payoff_t | 0.0000 | **yes** |
| baseline:random | baseline:always-pass | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | baseline:always-pass | accuracy_mcnemar | 1.0000 | no |
| baseline:random | baseline:oracle | payoff_t | 0.0000 | **yes** |
| baseline:random | baseline:oracle | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | baseline:oracle | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:random | baseline:oracle | ece_permutation | 0.0005 | **yes** |
| baseline:random | openrouter:openai_gpt-5.4-nano | payoff_t | 0.0000 | **yes** |
| baseline:random | openrouter:openai_gpt-5.4-nano | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | openrouter:openai_gpt-5.4-nano | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:random | openrouter:openai_gpt-5.4-nano | ece_permutation | 0.0005 | **yes** |
| baseline:random | openrouter:google_gemini-3.1-flash-lite | payoff_t | 0.0000 | **yes** |
| baseline:random | openrouter:google_gemini-3.1-flash-lite | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | openrouter:google_gemini-3.1-flash-lite | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:random | openrouter:google_gemini-3.1-flash-lite | ece_permutation | 0.0005 | **yes** |
| baseline:random | openrouter:google_gemini-3-flash-preview | payoff_t | 0.0000 | **yes** |
| baseline:random | openrouter:google_gemini-3-flash-preview | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | openrouter:google_gemini-3-flash-preview | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:random | openrouter:google_gemini-3-flash-preview | ece_permutation | 0.0005 | **yes** |
| baseline:random | openrouter:anthropic_claude-haiku-4.5 | payoff_t | 0.0000 | **yes** |
| baseline:random | openrouter:anthropic_claude-haiku-4.5 | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | openrouter:anthropic_claude-haiku-4.5 | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:random | openrouter:anthropic_claude-haiku-4.5 | ece_permutation | 0.0005 | **yes** |
| baseline:random | openrouter:meta-llama_llama-4-scout | payoff_t | 0.0000 | **yes** |
| baseline:random | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:random | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:random | openrouter:meta-llama_llama-4-scout | ece_permutation | 0.0005 | **yes** |
| baseline:always-take | baseline:always-pass | payoff_t | 0.0000 | **yes** |
| baseline:always-take | baseline:always-pass | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-take | baseline:always-pass | accuracy_mcnemar | 1.0000 | no |
| baseline:always-take | baseline:oracle | payoff_t | 0.0000 | **yes** |
| baseline:always-take | baseline:oracle | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-take | baseline:oracle | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-take | baseline:oracle | ece_permutation | 0.0005 | **yes** |
| baseline:always-take | openrouter:openai_gpt-5.4-nano | payoff_t | 0.0000 | **yes** |
| baseline:always-take | openrouter:openai_gpt-5.4-nano | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-take | openrouter:openai_gpt-5.4-nano | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-take | openrouter:openai_gpt-5.4-nano | ece_permutation | 0.0005 | **yes** |
| baseline:always-take | openrouter:google_gemini-3.1-flash-lite | payoff_t | 0.0000 | **yes** |
| baseline:always-take | openrouter:google_gemini-3.1-flash-lite | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-take | openrouter:google_gemini-3.1-flash-lite | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-take | openrouter:google_gemini-3.1-flash-lite | ece_permutation | 0.0005 | **yes** |
| baseline:always-take | openrouter:google_gemini-3-flash-preview | payoff_t | 0.0000 | **yes** |
| baseline:always-take | openrouter:google_gemini-3-flash-preview | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-take | openrouter:google_gemini-3-flash-preview | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-take | openrouter:google_gemini-3-flash-preview | ece_permutation | 0.0005 | **yes** |
| baseline:always-take | openrouter:anthropic_claude-haiku-4.5 | payoff_t | 0.0000 | **yes** |
| baseline:always-take | openrouter:anthropic_claude-haiku-4.5 | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-take | openrouter:anthropic_claude-haiku-4.5 | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-take | openrouter:anthropic_claude-haiku-4.5 | ece_permutation | 0.0005 | **yes** |
| baseline:always-take | openrouter:meta-llama_llama-4-scout | payoff_t | 0.0000 | **yes** |
| baseline:always-take | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-take | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-take | openrouter:meta-llama_llama-4-scout | ece_permutation | 0.0005 | **yes** |
| baseline:always-pass | baseline:oracle | payoff_t | 0.0000 | **yes** |
| baseline:always-pass | baseline:oracle | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-pass | baseline:oracle | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-pass | openrouter:openai_gpt-5.4-nano | payoff_t | 0.0000 | **yes** |
| baseline:always-pass | openrouter:openai_gpt-5.4-nano | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-pass | openrouter:openai_gpt-5.4-nano | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-pass | openrouter:google_gemini-3.1-flash-lite | payoff_t | 0.0000 | **yes** |
| baseline:always-pass | openrouter:google_gemini-3.1-flash-lite | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-pass | openrouter:google_gemini-3.1-flash-lite | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-pass | openrouter:google_gemini-3-flash-preview | payoff_t | 0.0000 | **yes** |
| baseline:always-pass | openrouter:google_gemini-3-flash-preview | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-pass | openrouter:google_gemini-3-flash-preview | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-pass | openrouter:anthropic_claude-haiku-4.5 | payoff_t | 0.0000 | **yes** |
| baseline:always-pass | openrouter:anthropic_claude-haiku-4.5 | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-pass | openrouter:anthropic_claude-haiku-4.5 | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:always-pass | openrouter:meta-llama_llama-4-scout | payoff_t | 0.0000 | **yes** |
| baseline:always-pass | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:always-pass | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:oracle | openrouter:openai_gpt-5.4-nano | payoff_t | 0.0000 | **yes** |
| baseline:oracle | openrouter:openai_gpt-5.4-nano | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:oracle | openrouter:openai_gpt-5.4-nano | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:oracle | openrouter:openai_gpt-5.4-nano | ece_permutation | 0.0005 | **yes** |
| baseline:oracle | openrouter:google_gemini-3.1-flash-lite | payoff_t | 0.0000 | **yes** |
| baseline:oracle | openrouter:google_gemini-3.1-flash-lite | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:oracle | openrouter:google_gemini-3.1-flash-lite | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:oracle | openrouter:google_gemini-3.1-flash-lite | ece_permutation | 0.0005 | **yes** |
| baseline:oracle | openrouter:google_gemini-3-flash-preview | payoff_t | 0.0000 | **yes** |
| baseline:oracle | openrouter:google_gemini-3-flash-preview | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:oracle | openrouter:google_gemini-3-flash-preview | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:oracle | openrouter:google_gemini-3-flash-preview | ece_permutation | 0.0090 | no |
| baseline:oracle | openrouter:anthropic_claude-haiku-4.5 | payoff_t | 0.0000 | **yes** |
| baseline:oracle | openrouter:anthropic_claude-haiku-4.5 | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:oracle | openrouter:anthropic_claude-haiku-4.5 | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:oracle | openrouter:anthropic_claude-haiku-4.5 | ece_permutation | 0.0005 | **yes** |
| baseline:oracle | openrouter:meta-llama_llama-4-scout | payoff_t | 0.0000 | **yes** |
| baseline:oracle | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.0000 | **yes** |
| baseline:oracle | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.0000 | **yes** |
| baseline:oracle | openrouter:meta-llama_llama-4-scout | ece_permutation | 0.0005 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3.1-flash-lite | payoff_t | 0.0051 | no |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3.1-flash-lite | payoff_wilcoxon | 0.4486 | no |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3.1-flash-lite | accuracy_mcnemar | 0.0009 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3.1-flash-lite | ece_permutation | 0.3193 | no |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3-flash-preview | payoff_t | 0.0000 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3-flash-preview | payoff_wilcoxon | 0.0000 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3-flash-preview | accuracy_mcnemar | 0.0000 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:google_gemini-3-flash-preview | ece_permutation | 0.0010 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:anthropic_claude-haiku-4.5 | payoff_t | 0.0000 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:anthropic_claude-haiku-4.5 | payoff_wilcoxon | 0.0000 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:anthropic_claude-haiku-4.5 | accuracy_mcnemar | 0.0000 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:anthropic_claude-haiku-4.5 | ece_permutation | 0.0025 | **yes** |
| openrouter:openai_gpt-5.4-nano | openrouter:meta-llama_llama-4-scout | payoff_t | 0.1361 | no |
| openrouter:openai_gpt-5.4-nano | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.2029 | no |
| openrouter:openai_gpt-5.4-nano | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.6655 | no |
| openrouter:openai_gpt-5.4-nano | openrouter:meta-llama_llama-4-scout | ece_permutation | 0.5857 | no |
| openrouter:google_gemini-3.1-flash-lite | openrouter:google_gemini-3-flash-preview | payoff_t | 0.0000 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:google_gemini-3-flash-preview | payoff_wilcoxon | 0.0000 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:google_gemini-3-flash-preview | accuracy_mcnemar | 0.0000 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:google_gemini-3-flash-preview | ece_permutation | 0.0005 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:anthropic_claude-haiku-4.5 | payoff_t | 0.0000 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:anthropic_claude-haiku-4.5 | payoff_wilcoxon | 0.0000 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:anthropic_claude-haiku-4.5 | accuracy_mcnemar | 0.0002 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:anthropic_claude-haiku-4.5 | ece_permutation | 0.0005 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:meta-llama_llama-4-scout | payoff_t | 0.2461 | no |
| openrouter:google_gemini-3.1-flash-lite | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.2876 | no |
| openrouter:google_gemini-3.1-flash-lite | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.0012 | **yes** |
| openrouter:google_gemini-3.1-flash-lite | openrouter:meta-llama_llama-4-scout | ece_permutation | 0.0240 | no |
| openrouter:google_gemini-3-flash-preview | openrouter:anthropic_claude-haiku-4.5 | payoff_t | 0.1955 | no |
| openrouter:google_gemini-3-flash-preview | openrouter:anthropic_claude-haiku-4.5 | payoff_wilcoxon | 0.8741 | no |
| openrouter:google_gemini-3-flash-preview | openrouter:anthropic_claude-haiku-4.5 | accuracy_mcnemar | 0.0352 | no |
| openrouter:google_gemini-3-flash-preview | openrouter:anthropic_claude-haiku-4.5 | ece_permutation | 0.3723 | no |
| openrouter:google_gemini-3-flash-preview | openrouter:meta-llama_llama-4-scout | payoff_t | 0.0000 | **yes** |
| openrouter:google_gemini-3-flash-preview | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.0000 | **yes** |
| openrouter:google_gemini-3-flash-preview | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.0000 | **yes** |
| openrouter:google_gemini-3-flash-preview | openrouter:meta-llama_llama-4-scout | ece_permutation | 0.0005 | **yes** |
| openrouter:anthropic_claude-haiku-4.5 | openrouter:meta-llama_llama-4-scout | payoff_t | 0.0000 | **yes** |
| openrouter:anthropic_claude-haiku-4.5 | openrouter:meta-llama_llama-4-scout | payoff_wilcoxon | 0.0000 | **yes** |
| openrouter:anthropic_claude-haiku-4.5 | openrouter:meta-llama_llama-4-scout | accuracy_mcnemar | 0.0000 | **yes** |
| openrouter:anthropic_claude-haiku-4.5 | openrouter:meta-llama_llama-4-scout | ece_permutation | 0.0005 | **yes** |