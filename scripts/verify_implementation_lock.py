import sys

from validation.verify_lock import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--kind", "implementation"]
    raise SystemExit(main())
