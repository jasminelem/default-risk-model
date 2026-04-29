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
import pandas as pd


def _read_pgpass_username():
    """Extract the WRDS username from ~/.pgpass if it exists."""
    pgpass = Path.home() / ".pgpass"
    if not pgpass.exists():
        return None
    for line in pgpass.read_text().splitlines():
        parts = line.strip().split(":")
        if len(parts) >= 4 and "wrds" in parts[0]:
            return parts[3]
    return None


def get_wrds_connection(verbose: bool = True) -> "wrds.Connection":
    """
    Returns a live WRDS connection.

    Resolution order for credentials:
      1. WRDS_USERNAME env var  +  ~/.pgpass for password
      2. Username parsed from ~/.pgpass
      3. Fails with a clear error (never prompts for input)
    """
    try:
        import wrds
    except ImportError:
        raise RuntimeError(
            "The 'wrds' package is not installed.\n"
            "Run: pip install wrds"
        )

    username = os.environ.get("WRDS_USERNAME") or _read_pgpass_username()
    if not username:
        raise RuntimeError(
            "Could not determine WRDS username.\n"
            "Either set the WRDS_USERNAME environment variable, or ensure\n"
            "~/.pgpass contains a line like:\n"
            "  wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD\n"
            "Then run:  chmod 600 ~/.pgpass"
        )

    if verbose:
        print(f"Connecting to WRDS as '{username}' (via ~/.pgpass)...")

    try:
        conn = wrds.Connection(wrds_username=username, autoconnect=True)
        _ = conn.raw_sql("SELECT 1 as ok")
        if verbose:
            print("WRDS connection successful.")
        return conn
    except Exception as e:
        msg = str(e).lower()
        if "password" in msg or "authentication" in msg or "pgpass" in msg or "fe_sendauth" in msg:
            raise RuntimeError(
                f"\nWRDS authentication failed for user '{username}'.\n\n"
                "Make sure ~/.pgpass has the correct password:\n"
                "  wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD\n\n"
                "Then run:  chmod 600 ~/.pgpass\n"
            ) from e
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