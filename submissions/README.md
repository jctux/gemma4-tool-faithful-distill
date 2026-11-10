# Exact file sets of every hidden-task submission

Each directory is the unzipped `submission.zip` as uploaded to the Kaggle main track
(`agent.yaml`, `eval_config.yaml`, `configs/sampling.yaml`, `prompts/system.md`).
Scores are the public leaderboard split (about 58 of the ~120 hidden tasks; one task = 0.017).

| dir | Kaggle id | model | budgets (calls / min / turns) | thinking | prompt | public score |
|---|---|---|---|---|---|---|
| `e4b_base` | 57679047 | E4B | 40 / 8 / 60 | 1024 | harness-tuned baseline | **0.05** |
| `p2` | 57721934 | E4B | 40 / 8 / 60 | 1024 | baseline + prompt2 rules | 0.03 |
| `31b` | 57753557 | 31B | 40 / 8 / 60 | 1024 | harness-tuned baseline | **0.13** |
| `31b_think` | 57813550 | 31B | 40 / 8 / 60 | 4096 | harness-tuned baseline | Notebook Timeout |
| `31b_d2` | 57850581 | 31B | 60 / 12 / 90 | 1024 | harness-tuned baseline | Notebook Timeout |
| `v6` | 57908574 | 31B | 28 / 8 / 80 | 4096 | public-leader prompt (flexonafft) | Notebook Timeout |
| `v6t1` | 57961478 | 31B | 28 / 8 / 80 | 1024 | public-leader prompt (flexonafft) | 0.12 |
| `e4b_v6` | 57999821 | E4B | 28 / 8 / 80 | 1024 | public-leader prompt (flexonafft) | 0.05 |
| `budgetfit` | 58025117 | 31B | 28 / 8 / 80 | 4096 | Budget-Fit (verracodeguacas) | **0.13** |
| `e4b_bf` | 58066052 | E4B | 28 / 8 / 80 | 4096 | Budget-Fit (verracodeguacas) | 0.03 |
| `bf40` | 58111072 | 31B | 40 / 12 / 80 | 4096 | Budget-Fit, budget numbers updated | Notebook Timeout |
| `bf34` | not submitted | 31B | 34 / 10 / 80 | 4096 | Budget-Fit, budget numbers updated | - |

No submission contains a LoRA adapter: every adapter we trained scored below the base model on
the held-out tasks (see the writeup).

Prompt credits: `v6*`/`e4b_v6` use the system prompt from the public notebook
`gemma-4-baseline-lb-0-24` (flexonafft, "budget-aware agent v2"); `budgetfit`, `e4b_bf`, `bf40`,
`bf34` use the public "Budget-Fit single-agent" file set by verracodeguacas (Apache 2.0),
repackaged in `leoprovorov/score-to-gemma-4`. Both are used unchanged except where the table says
budget numbers were updated.
