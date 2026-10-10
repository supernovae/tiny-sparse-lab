# dense-lm-scale-v1 preregistration

Recorded before any new scale-study staging or training. `protocol.md` and all three scale configs are frozen by the raw SHA-256 digests below; put outcomes and deviations in `results.md` without editing frozen inputs. The 16.78M small reference is not a newly promoted checkpoint.

| Input | SHA-256 |
|---|---|
| `experiments/research/dense-lm-scale-v1/protocol.md` | `b0cbd81dceaa9757cf0858e4252c775b2bb26087b8c536266e17b63c501f63dc` |
| `configs/dense_lm_scale_v1_seed42.yaml` | `64a6b82f7e82521f1ceb577b1f7a5d7fee5b6aad32f224bec046e9a171433713` |
| `configs/dense_lm_scale_v1_seed17.yaml` | `eed534179a1c38698885a9784347d3305ac81a8cf5dbbf23e8aed517a155a742` |
| `configs/dense_lm_scale_v1_seed73.yaml` | `f90fe0a082e4bd3900973a87f959c1818c335225f979abdfb361a227997338d9` |
| `data/dense_lm_v1_prompts.json` | `118839f291b8741d233ce19baff15f0dcb9cef52eb907208ff6ac9a81bc3408e` |
| `artifacts/acceptance/dense_lm_token_budget_v1.json` | `41ddc5be2290bbd568419adc9786bb1b00483623598c2926e130f4919593e51b` |
| `artifacts/acceptance/dense_lm_v1.json` | `f59fd9e3a068202c6690ea0a5357887f251d8ed84cc37871023f4632fbcb0dfa` |

Original promoted checkpoint manifest digests at 4,096: seed 42 `53376abc14382e38b72a3f41efa2cf3aaacbbd1f6e3b67cf91499ad87d5bf816`, 17 `781f2a2da6b4c82a7974b69b326c2d66830bc2a385fdffd60e16440bced0c0e3`, 73 `cd248c47746f928bd11c6e5588e4f46449d292e63e79bd59c06fb7ddb7ab9013`. The parent reference at 16,384 is bound in the accepted token-budget JSON; its seed-42 checkpoint digest is `c4ef91e5cd5e806494ca7523182e00182d249a639e6d026712494098907934b4`, seed 17 `1e8d6181e34aa3572771358c3021187a740aaab5b54f4d4478da754fc59b0e08`, seed 73 `fe37fd2b1c8cdf62f42a104a17a0f77f70d39ce45a09fe17f8227f726d37e383`.

Existing tokenizer SHA-256 `6bd2b7c7086c85bcac66ba5d17a2f822805fab9f1c6fe7973d8569f4fb22b44e`, train array `1e9eb838ee865c5cbb9bc869e3c3dc4d00c1b908f432f9e84dd605d974339a46`, validation array `3dac074c832502f92cf00d516c4fb83b5732b8e651e84b658fa7dfe69f4a56fd`, train supervision `384cbed5e1d3763f0338dfef374936e3130c5a9f287c74707b95f18048cce247`, validation supervision `39ba468be18ac145836eceff8fa05d8f7e93f6225a6617ecd2821aa387391ddd` (checked against run-owned files). Prior source revision: git `4fe0e4719f40ac727592803e2debff4963acb91b`; current study's new config and documentation change the source identity recorded in any new run. Original vendor device: RX 7900 XTX; do not infer its availability from historical runs. No training run is accepted without live ROCm verification.
