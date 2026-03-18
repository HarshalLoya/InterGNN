#!/usr/bin/env python3
"""
Git Auto-Commit Script for ARGUS

This script automatically commits and pushes each untracked and modified file
individually with descriptive commit messages. It recursively scans ALL folders
and subfolders in the repository.

Usage:
    python git_auto_commit.py              # Dry run (show what would be committed)
    python git_auto_commit.py --push       # Commit and push each file
    python git_auto_commit.py --commit     # Commit only, no push
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        print(f"❌ Git error: {result.stderr}")
    return result


def get_all_changes() -> dict:
    """
    Get ALL changes in the repository recursively.
    Scans all folders, subfolders, and files.
    
    Returns dict with keys: 'untracked', 'modified', 'deleted', 'staged'
    """
    changes = {
        'untracked': [],
        'modified': [],
        'deleted': [],
        'staged': [],
    }
    
    # Get untracked files (recursively scans all directories)
    result = run_git(["ls-files", "--others", "--exclude-standard"], check=False)
    if result.returncode == 0 and result.stdout.strip():
        changes['untracked'] = [f for f in result.stdout.strip().split("\n") if f]
    
    # Get modified files (not staged) - recursive
    result = run_git(["diff", "--name-only"], check=False)
    if result.returncode == 0 and result.stdout.strip():
        changes['modified'] = [f for f in result.stdout.strip().split("\n") if f]
    
    # Get deleted files - recursive
    result = run_git(["diff", "--name-only", "--diff-filter=D"], check=False)
    if result.returncode == 0 and result.stdout.strip():
        changes['deleted'] = [f for f in result.stdout.strip().split("\n") if f]
    
    # Get staged files - recursive
    result = run_git(["diff", "--cached", "--name-only"], check=False)
    if result.returncode == 0 and result.stdout.strip():
        changes['staged'] = [f for f in result.stdout.strip().split("\n") if f]
    
    # Remove deleted from modified (tracked separately)
    changes['modified'] = [f for f in changes['modified'] if f not in changes['deleted']]
    
    return changes


def scan_directory_tree() -> None:
    """Print directory tree being scanned for visibility."""
    print("📂 Scanning directory tree recursively...")
    result = run_git(["ls-files", "--others", "--exclude-standard", "--directory"], check=False)
    
    # Count directories
    dirs_scanned = set()
    for f in Path(".").rglob("*"):
        if f.is_dir() and ".git" not in str(f):
            dirs_scanned.add(str(f))
    
    print(f"   Found {len(dirs_scanned)} directories to scan")


def generate_commit_message(filepath: str, action: str) -> str:
    """
    Generate a descriptive commit message with filename prefix.
    Format: "filename: Action full/path/to/file"
    """
    path = Path(filepath)
    filename = path.name
    
    # Action prefix
    if action == "new":
        action_word = "Add"
    elif action == "modified":
        action_word = "Update"
    elif action == "deleted":
        action_word = "Remove"
    elif action == "staged":
        action_word = "Update"
    else:
        action_word = "Update"
    
    # Format: "filename: Action full/path/to/file"
    return f"{filename}: {action_word} {filepath}"


def commit_file(filepath: str, action: str, push: bool = False) -> bool:
    """Commit a single file with a descriptive message."""
    message = generate_commit_message(filepath, action)
    
    # Stage the file
    if action == "deleted":
        result = run_git(["rm", filepath], check=False)
    else:
        result = run_git(["add", filepath], check=False)
    
    if result.returncode != 0:
        print(f"  ❌ Failed to stage: {filepath}")
        return False
    
    # Commit
    result = run_git(["commit", "-m", message], check=False)
    if result.returncode != 0:
        # Check if nothing to commit
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            print(f"  ⚠️  Already committed: {filepath}")
            return True
        print(f"  ❌ Failed to commit: {filepath}")
        print(f"     {result.stderr}")
        return False
    
    print(f"  ✅ Committed: {filepath}")
    print(f"     Message: {message}")
    
    # Push if requested
    if push:
        result = run_git(["push"], check=False)
        if result.returncode != 0:
            print(f"  ⚠️  Push failed: {result.stderr}")
            return False
        print(f"  🚀 Pushed")
    
    return True


def process_all_files(changes: dict, dry_run: bool, push: bool) -> tuple[int, int]:
    """Process all files in all categories."""
    success = 0
    failed = 0
    
    # Process in order: staged -> modified -> untracked -> deleted
    categories = [
        ("staged", "📋 Processing staged files..."),
        ("modified", "📝 Processing modified files..."),
        ("untracked", "📁 Processing untracked files..."),
        ("deleted", "🗑️  Processing deleted files..."),
    ]
    
    for category, message in categories:
        files = changes.get(category, [])
        if not files:
            continue
            
        print()
        print(message)
        
        for filepath in sorted(files):
            if dry_run:
                action_symbol = {"untracked": "+", "modified": "M", "deleted": "D", "staged": "S"}.get(category, "?")
                msg = generate_commit_message(filepath, category)
                print(f"   {action_symbol} {filepath}")
                print(f"     → {msg}")
            else:
                if commit_file(filepath, category, push):
                    success += 1
                else:
                    failed += 1
    
    return success, failed


def main():
    # Parse arguments
    dry_run = True
    push = False
    
    if "--push" in sys.argv:
        dry_run = False
        push = True
    elif "--commit" in sys.argv:
        dry_run = False
        push = False
    
    print("=" * 60)
    print("🔍 ARGUS Git Auto-Commit Script v1.1")
    print("   Recursively scans ALL folders and subfolders")
    print("=" * 60)
    print()
    
    # Check if we're in a git repo
    result = run_git(["status"], check=False)
    if result.returncode != 0:
        print("❌ Not a git repository!")
        return 1
    
    # Show what we're scanning
    scan_directory_tree()
    print()
    
    # Get ALL changes recursively
    changes = get_all_changes()
    
    total = sum(len(v) for v in changes.values())
    
    if total == 0:
        print("✨ No changes to commit!")
        return 0
    
    print(f"📊 Found {total} files to process:")
    print(f"   • Staged files: {len(changes['staged'])}")
    print(f"   • Modified files: {len(changes['modified'])}")
    print(f"   • New (untracked) files: {len(changes['untracked'])}")
    print(f"   • Deleted files: {len(changes['deleted'])}")
    
    if dry_run:
        print()
        print("🔍 DRY RUN - No changes will be made")
        print("   Use --commit to commit only")
        print("   Use --push to commit and push")
    
    # Process all files
    success, failed = process_all_files(changes, dry_run, push)
    
    if not dry_run:
        print()
        print("=" * 60)
        print(f"✅ Completed: {success} successful, {failed} failed")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print(f"📋 Summary: {total} files would be committed")
        print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
