"""Invocation-local acquisition hooks; no network, decoder or runtime imports."""

from contextvars import ContextVar

# Native callers outside an admitted portable invocation retain their contracts.
current_transport = ContextVar("sparselab_dataset_transport", default=None)
current_verifications = ContextVar("sparselab_dataset_verifications", default=())


def verification_transport(path):
    for transport in current_verifications.get():
        if path.parent in (transport.output, transport.publication):
            return transport
    return None
