"""Check optional backend compatibility before expensive training/evaluation."""
from importlib.metadata import PackageNotFoundError, version


def main():
    for name in ("torch", "transformers", "peft", "torchao", "bitsandbytes"):
        try:
            print(f"{name}=={version(name)}", flush=True)
        except PackageNotFoundError:
            print(f"{name}: absent", flush=True)
    from peft.import_utils import is_torchao_available
    try:
        is_torchao_available()
    except ImportError as exc:
        raise RuntimeError(
            "Optional torchao installation is incompatible with PEFT. "
            "This project uses bitsandbytes NF4, not torchao. In the dedicated "
            "Kaggle training environment, run `python -m pip uninstall -y torchao` "
            "and rerun this check in a fresh Python process. Do not retrain an "
            "existing adapter just to repair evaluation dependencies."
        ) from exc
    print("PEFT optional backend compatibility passed", flush=True)


if __name__ == "__main__":
    main()
