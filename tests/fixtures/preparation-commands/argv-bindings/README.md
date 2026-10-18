# Neutral preparation argument fixtures

The 25-phase plan covers the canonical chain's 17 native preparation/inspection
command shapes. It exercises source-relative defaults, cross-phase argument and
template references, nested source leaves, relative template bindings, directory
outputs, disjoint receipts and logs, and measurement-only phase selection.

These are command compilation fixtures, not an executable corpus recipe. Input
releases, inventories, policy documents, source bytes and release declarations
are deliberately absent. Placeholder JSON/YAML templates demonstrate rendering;
they do not claim scientific schema validity or approval. Tests render only tiny
project YAML; they never acquire data, build corpora or train models. Bounds are
fixture values, not research recommendations.

The generic test names all 17 expected command shapes so removing a phase cannot
silently narrow coverage. It also checks exact resolved paths and refusals for
invalid references, duplicate leaves, mismatched outputs and unresolved bindings.
Scientific KML declarations and specialized tests remain separate and unchanged,
except that their two historical measurement hash assertions now live in the
specialized suite while the originals remain in Lab.
