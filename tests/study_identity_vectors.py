"""Fixed neutral report identities shared by reader and acceptance tests."""

# Recorded from the existing neutral report fixture at Lab 2a0342de653b766f87b2297e12c4fb9f741bfb6d.
# Fixed expectations intentionally do not come from the reader under test.
# Intentional smoke-config/card changes may justify reviewed vector regeneration.
# Additive RunConfig defaults that change old receipt identities are a normalization
# regression: preserve the historical guarantee, do not merely regenerate hashes.
MATRIX = "e1e4b9eae75254117f0d1c2b7ec7a8deaf075f1d196d467d1eeb2fcc31359bfc"
STUDY = "ec13286d47d1d4f770940f415f7caa90f01bb560b6a781d12d7f578c799b51dc"
CONFIGS = [
    "0a2c6cff8756684c12f1ce2743eb392e40a161ce64caa14dbd039aeaa56a4787",
    "7299f564f23a6493afe3bcc2ec187d731028ea7aa1fdc45ff5a6331dfe140df8",
]
