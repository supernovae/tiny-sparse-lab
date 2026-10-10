# dense-lm-token-budget-v1 preregistration

Recorded **before any budget-extension training**. These SHA-256 digests bind the frozen [protocol](protocol.md), configs and original controls; do not revise this record after seeing outcomes. Execution outputs and decisions go to `results.md`.

| Input | SHA-256 |
|---|---|
| `experiments/research/dense-lm-token-budget-v1/protocol.md` | `28faabdda24b046b11a0c0c64f2587375703199c479fd8b3ba33052223ea7483` |
| `configs/dense_lm_token_budget_v1_seed42.yaml` | `6f04259e478eba90f7dc71e54102555a81b59094d543f3e6de1ab06a464c8828` |
| `configs/dense_lm_token_budget_v1_seed17.yaml` | `19d242150edcee119ed0d82e8a9398b07b92a72a7c30ccb7c2b3951ff2c4126d` |
| `configs/dense_lm_token_budget_v1_seed73.yaml` | `573072284cdcf6a7ff060a999c7e58d790319f791052b28e240eb19547ee0c35` |
| `data/dense_lm_v1_prompts.json` | `118839f291b8741d233ce19baff15f0dcb9cef52eb907208ff6ac9a81bc3408e` |
| seed 42 terminal parent `checkpoints/step_00004096_gen_000007/manifest.json` | `53376abc14382e38b72a3f41efa2cf3aaacbbd1f6e3b67cf91499ad87d5bf816` |
| seed 17 terminal parent `checkpoints/step_00004096_gen_000007/manifest.json` | `781f2a2da6b4c82a7974b69b326c2d66830bc2a385fdffd60e16440bced0c0e3` |
| seed 73 terminal parent `checkpoints/step_00004096_gen_000007/manifest.json` | `cd248c47746f928bd11c6e5588e4f46449d292e63e79bd59c06fb7ddb7ab9013` |

Each parent is under `sparselab-work/experiments/dense-lm-v1/runs/dense-lm-v1-seed<SEED>-step4096/`. Their checkpoint manifest digests above match the [accepted chains](../../../artifacts/acceptance/dense_lm_v1.json): seed 42 `b6e77e30cc78731a74dd713471106c4b10afca2901da501b95d30e45c1b1b524`, seed 17 `2f50048fb000ad9df616fa2cd82c0337da328d057b2fd867dd893e9ed681146c`, seed 73 `55bfbca54566d39fac7020d7cc6cdb46159ec207887740345fa25693cf63c26b`.

Original baseline source identity SHA-256: `927dca9db1746ce2909013763980c7403b0b165ac59ef559a1f78b0eac4d6650` ([lifecycle](../../../configs/dense_lm_v1_lifecycle.json)); source revision used for this new study is local `main` commit `a5bd982` plus the new budget-extension implementation, whose run manifest records the exact effective source identity. Original pinned TinyStories revision is `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The pinned tokenizer SHA-256 is `6bd2b7c7086c85bcac66ba5d17a2f822805fab9f1c6fe7973d8569f4fb22b44e`, and the retained seed-42 training and validation arrays hash to `1e9eb838ee865c5cbb9bc869e3c3dc4d00c1b908f432f9e84dd605d974339a46` and `3dac074c832502f92cf00d516c4fb83b5732b8e651e84b658fa7dfe69f4a56fd` respectively. The original configs, panel and baseline acceptance stay unchanged.
