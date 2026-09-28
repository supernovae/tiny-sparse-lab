# Hugging Face access

SparseLab uses Hugging Face Hub for Corpus Forge `huggingface_dataset` acquisition, explicit `https://huggingface.co/...` HTTP documents, remote TinyStories/FineWeb-Edu/Cosmopedia streams, the one-time local TinyStories snapshot, and the pinned Pythia reference-model download. Configure credentials **before** launching `uv run --locked sparselab ...` so these requests use your account rather than an anonymous Hub session.

## Set up a token

Create a **read** token at [Hugging Face token settings](https://huggingface.co/settings/tokens). For gated repositories, accept the repository's terms with that account first; a token alone does not grant access. Choose either:

```sh
# Recommended for an interactive workstation: stored in your user account's Hub cache.
uv run --locked hf auth login
uv run --locked hf auth whoami
```

Alternatively, set `HF_TOKEN` in the **process environment before launch**, for
example through a secret manager. The Hub reads environment variables at import
time, so setting one inside an already running Python session is too late. For
one explicit CLI invocation, supply a token **file path**, not the token itself:

```sh
uv run --locked sparselab --hf-token-file "$HOME/.config/sparselab/hf-token" \
  corpus acquire corpora/my-corpus/corpus.yaml
```

The file must contain a nonblank token; restrict its permissions to the intended
user, or use your worker's mounted secret facility. Precedence is
`--hf-token-file` > `HF_TOKEN` > locally saved `hf auth login`. To check the
session's environment token or login without printing the secret:

```sh
uv run --locked hf auth whoami
```

Avoid putting tokens in YAML, source declarations, shell history, literal CLI
arguments, checked-in files, or run artifacts. A literal `--hf-token VALUE`
option is intentionally absent: process listings and shell history expose it.
Local and SSH workers, containers and scheduled jobs need their own
credentials/environment; a login on the controller does not authenticate a
remote worker. If you relocate the Hub credential store with `HF_HOME`, set that
variable for both login and SparseLab. `--work-dir`/`SPARSELAB_WORK_DIR` control
SparseLab's scratch space, **not** the Hub login store or `HF_TOKEN`.

## Behavior and boundaries

For an authenticated network request, SparseLab forwards the token selected
from the explicit file, `HF_TOKEN`, or locally saved Hub login to the
`datasets`/`huggingface_hub` client. For a Corpus Forge HTTP document, it sends
a bearer token **only** to the exact HTTPS `huggingface.co` host and removes it
on a redirect to another origin; generic HTTP sources never receive it.
Without any configured token it retains anonymous public access. Token bytes
are neither printed nor placed in corpus declarations, snapshot/build/release
identities, exported configs, tokenizer/prepared manifests, or run evidence.
Switching credentials does not change a pinned dataset/model revision or an
immutable corpus release; changing source bytes or the pinned revision does.

When a Hub request needs credentials, an empty explicit token or a
missing/unreadable token file fails instead of falling back to anonymous access.
The Hub cannot establish whether a token is
valid without contacting it: a 401/403 response reports rejected credentials or
denied repository access, without showing the secret. Check `hf auth whoami` in
the same session, verify the token's read scope and accept gated-repository
terms. Authentication avoids **anonymous** limits but does not eliminate
account-level throttling, data-transfer limits or network failures. Do not paste
tokens into error reports. Corpus Forge's local sample and an already verified
`corpus build --offline` need no Hugging Face credentials or network. See the
[offline sample recipe](../corpora/devmind-sample-v0/README.md).
