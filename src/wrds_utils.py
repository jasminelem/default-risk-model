"""
src/wrds_utils.py

Central place for all WRDS connections in this project.

Goal: NEVER require the user to type a password interactively.

Assumption:
- You have already set up ~/.pgpass correctly (the standard WRDS way).
  Example line in ~/.pgpass:
  wrds.wrds.wharton.upenn.edu:5432:your_db:your_wrds_username:your_password

  Then run:
      chmod 600 ~/.pgpass

After that, wrds.Connection() connects silently.

This module provides a single get_wrds_connection() function that all scripts
(build_company_index, ratings pull, etc.) should use.
"""

# Allow "from src.xxx" when the file is executed directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Optional
import pandas as pd


def get_wrds_connection(verbose: bool = True) -> "wrds.Connection":
    """
    Returns a live WRDS connection.

    - Tries to use existing ~/.pgpass (no password prompt).
    - If that fails with an auth error, gives a clear one-time instruction
      instead of hanging on a password prompt.
    """
    try:
        import wrds
    except ImportError:
        raise RuntimeError(
            "The 'wrds' package is not installed.\n"
            "Run: pip install wrds"
        )

    if verbose:
        print("Connecting to WRDS (using ~/.pgpass if configured)...")

    try:
        # First try the standard non-interactive path (relies on ~/.pgpass)
        conn = wrds.Connection(autoconnect=True)
        # Quick sanity check
        _ = conn.raw_sql("SELECT 1 as ok")
        if verbose:
            print("✓ WRDS connection successful (silent, using ~/.pgpass).")
        return conn
    except Exception as e:
        # If it still wants input, force a clear error instead of hanging
        if "get_user_credentials" in str(type(e)) or "input" in str(e).lower():
            raise RuntimeError(
                "WRDS is still trying to prompt for credentials.\n"
                "Please run this once in an interactive terminal:\n"
                "    python -c \"import wrds; wrds.Connection()\"\n"
                "and answer the prompts. After that the .pgpass file will be correct and all future runs (including this script) will be completely silent."
            ) from e
        raise

    except Exception as e:
        msg = str(e).lower()
        if "password" in msg or "authentication" in msg or "pgpass" in msg or "fe_sendauth" in msg:
            raise RuntimeError(
                "\nWRDS authentication failed.\n\n"
                "You said you already have the password saved in a file.\n"
                "Please make sure ~/.pgpass is correctly set up:\n\n"
                "1. Create/edit ~/.pgpass with this line (replace values):\n"
                "   wrds.wrds.wharton.upenn.edu:5432:your_db_name:YOUR_WRDS_USERNAME:YOUR_PASSWORD\n\n"
                "2. Set correct permissions:\n"
                "   chmod 600 ~/.pgpass\n\n"
                "3. Make sure the file is in your HOME directory (~).\n\n"
                "After that, this script (and build_company_index.py, etc.) will never ask for a password again.\n"
                "Full guide: https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python-python-from-your-computer/"
            ) from e
        else:
            # Some other connection error (network, etc.)
            raise RuntimeError(f"Could not connect to WRDS: {e}") from e


def test_connection() -> bool:
    """Quick test used in scripts."""
    try:
        conn = get_wrds_connection(verbose=False)
        conn.close()
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    # Allow the user to test their setup easily
    print("Testing your WRDS password-less setup...")
    if test_connection():
        print("\nPerfect — future runs of build_company_index.py and the ratings pull will be completely silent.")
    else:
        print("\nPlease fix ~/.pgpass as described above.")