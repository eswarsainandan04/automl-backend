"""
Standalone runner for AutoGluon Missing Value Handler.

Called by the preprocessing pipeline via subprocess using the configured
Python executable.

Usage:
    python missing_values_runner.py <user_id> <session_id>
"""
import os
import sys

# Add this directory (data_preprocessing/) to sys.path so that bare imports
# in missing_values_handler.py (e.g. `from supabase_storage import ...`) resolve.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Also add the backend root so config / other sibling packages are available.
_BACKEND_ROOT = os.path.dirname(_THIS_DIR)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from missing_values_handler import AutoGluonMissingValueHandler


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python missing_values_runner.py <user_id> <session_id>",
            file=sys.stderr,
        )
        sys.exit(1)

    user_id    = sys.argv[1]
    session_id = sys.argv[2]

    print("=" * 70)
    print("  Missing Values Runner (AutoGluon venv)")
    print("=" * 70)
    print(f"User ID:    {user_id}")
    print(f"Session ID: {session_id}")
    print()

    handler = AutoGluonMissingValueHandler(user_id=user_id, session_id=session_id)
    stats   = handler.process_all_datasets()

    print("\nRunner finished.")
    if stats:
        print(f"  Files processed : {stats.get('files_processed', '?')}")
        print(f"  Missing before  : {stats.get('total_missing_before', '?')}")
        print(f"  Missing after   : {stats.get('total_missing_after', '?')}")


if __name__ == "__main__":
    main()
