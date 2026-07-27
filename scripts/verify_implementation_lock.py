from validation.verify_lock import main
import sys
if __name__ == "__main__":
    sys.argv[1:1] = ["--kind", "implementation"]
    raise SystemExit(main())
