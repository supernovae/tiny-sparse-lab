"""Framework-free entry point for operational environment management."""

import sys


def main() -> None:
    if sys.argv[1:3] == ["runtime", "env"]:
        from sparselab.runtime_env_cli import main as environment_main

        environment_main(sys.argv[3:])
    else:
        from sparselab.cli.main import main as application_main

        application_main()
