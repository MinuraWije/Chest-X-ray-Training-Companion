"""Run the MedGemma notebook server as a module.

    python -m kaggle_runtime.medgemma_server
    python -m kaggle_runtime.medgemma_server --port 9000
"""
from .medgemma_server import main

if __name__ == "__main__":
    main()
