"""Framework-free entry point for operational environment management."""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    # REMAINDER keeps command-local flags out of global prefix selection.
    prefix = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    prefix.add_argument("--work-dir", type=Path)
    prefix.add_argument("command", nargs=argparse.REMAINDER)
    args, extras = prefix.parse_known_args(sys.argv[1:])
    if not extras and args.command[:2] == ["runtime", "env"]:
        from sparselab.runtime_env_cli import main as environment_main
        from sparselab.workdir import resolve_work_dir

        if args.work_dir is not None:
            os.environ["SPARSELAB_WORK_DIR"] = str(resolve_work_dir(args.work_dir))
        environment_main(args.command[2:])
    elif not extras and args.command[:1] == ["lab-job"]:
        from sparselab.lab_job import main as job_main

        if (
            args.work_dir is not None
            or getattr(args, "hf_token_file", None) is not None
        ):
            prefix.error(
                "lab-job paths come only from its immutable manifest; global work/token options are unsupported"
            )
        job_main(args.command[1:])
    else:
        from sparselab.cli.main import main as application_main

        application_main()
