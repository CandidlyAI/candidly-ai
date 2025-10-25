#!/usr/bin/env python3
"""
main.py
-------
1) Runs onboarding (01_onboarding.run_onboarding).
2) Saves the result to data/completes.json.
3) Reads that JSON and starts the main conversation loop from 02_main_loop.py
   using functions that accept the onboarding dict.
"""
import json, os, sys
from pathlib import Path
import importlib.util

THIS_DIR = Path(__file__).parent
SRC_ONBOARD = THIS_DIR / "01_onboarding.py"
SRC_LOOP    = THIS_DIR / "02_main_loop.py"
OUT_JSON    = THIS_DIR / "data" / "completes.json"

def import_by_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    # Ensure output directory
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Import onboarding
    if not SRC_ONBOARD.exists():
        print(f"[error] Missing 01_onboarding.py at {SRC_ONBOARD}", file=sys.stderr); sys.exit(2)
    onboarding_mod = import_by_path(SRC_ONBOARD, "onboarding_mod")

    if not hasattr(onboarding_mod, "run_onboarding"):
        print("[error] 01_onboarding.py does not expose run_onboarding()", file=sys.stderr); sys.exit(3)

    # Run onboarding and save
    result = onboarding_mod.run_onboarding()
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Wrote completes JSON to: {OUT_JSON}")

    # Read back JSON (explicitly doing what you asked)
    data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    # Basic validation
    for k in ("role","ai_role","is_done"):
        if k not in data:
            print(f"[warn] Onboarding JSON missing key: {k}")

    # Import conversation loop module
    if not SRC_LOOP.exists():
        print(f"[error] Missing 02_main_loop.py at {SRC_LOOP}", file=sys.stderr); sys.exit(4)
    loop_mod = import_by_path(SRC_LOOP, "loop_mod")

    # Ensure functions exist
    if not hasattr(loop_mod, "run_conversation_from_onboarding"):
        print("[error] 02_main_loop.py does not expose run_conversation_from_onboarding(onboarding, ...)")
        sys.exit(5)

    # Optional knobs via env; defaults: 3 turns, 5 seconds per recording
    num_turns = int(os.getenv("CONVO_TURNS", "3"))
    rec_secs  = int(os.getenv("CONVO_REC_SEC", "5"))

    # Launch the conversation
    convo_result = loop_mod.run_conversation_from_onboarding(data, num_turns=num_turns, recording_duration=rec_secs)

    # If you want to persist the conversation log, uncomment:
    # (THIS_DIR / "data" / "conversation_log.json").write_text(json.dumps(convo_result, ensure_ascii=False, indent=2), "utf-8")

if __name__ == "__main__":
    main()
