#!/usr/bin/env python3
"""E1 lifecycle CLI; use --help and a subcommand's --help for explicit stages."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e1.runner import main
if __name__=='__main__':main()
