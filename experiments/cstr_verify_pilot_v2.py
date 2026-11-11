"""Read-only archived-record verification, with real-operation entrypoints blocked."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cstr_pilot_review_v2 import main
if __name__=='__main__':main()
