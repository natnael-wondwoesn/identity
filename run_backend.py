#!/usr/bin/env python3
"""
Startup script for the Market Research API Backend
Run this from the project root directory
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    # Ensure we're running from the project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)

    # Add project root to Python path
    sys.path.insert(0, str(project_root))

    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_root)

    # Run the backend
    backend_path = project_root / "backend" / "main.py"

    if not backend_path.exists():
        print("Error: backend/main.py not found!")
        sys.exit(1)

    print(f"Starting Market Research API from {project_root}")
    print(f"Backend path: {backend_path}")

    try:
        # Run with uvicorn
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ],
            cwd=project_root,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting backend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
