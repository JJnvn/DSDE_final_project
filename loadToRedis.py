# load_cleanv3_to_redis.py
import csv, json, redis, pathlib, sys

CSV_PATH = pathlib.Path("cleanv3.csv")         # ⇦ adjust if the file lives elsewhere
KEY_PREFIX = "cleanv3"                         # keeps your keys separate
SET_NAME   = f"{KEY_PREFIX}:keys"              # an index set

def main():
    rd = redis.Redis(encoding="utf‑8", decode_responses=True)

    # speed‑up: do everything in one pipeline
    pipe = rd.pipeline()

    with CSV_PATH.open(newline="", encoding="utf‑8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            k = f"{KEY_PREFIX}:{idx}"
            pipe.set(k, json.dumps(row, ensure_ascii=False))
            pipe.sadd(SET_NAME, k)

    pipe.execute()
    print(f"✔  Loaded {idx+1} rows into Redis under prefix '{KEY_PREFIX}:'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
