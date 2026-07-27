"""Deprecated compatibility entrypoint: verifies both locks that exist."""
from pathlib import Path
from validation.verify_lock import DEFAULT_LOCKS, verify

errors=[]
for kind,path in DEFAULT_LOCKS.items():
    if path.exists(): errors.extend(verify(kind,path))
if errors:
    print("MISMATCH")
    print("\n".join(errors))
    raise SystemExit(2)
print("VERIFIED")
