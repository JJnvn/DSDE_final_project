"""
Pipeline stages
===============
1. **Website scrape** – reuse `scrape_fondue()` from *fondue_scraper.py* to pull
   finished Bangkok cases directly from the Traffy Fondue web app (uses
   Selenium).
2. **Cleaning** – run the same `fix_type()` / `fix_org()` logic from
   *cleanScape.py* and calculate `eta_hours`, `log_eta`, …
3. **API pull** – call the official JSON API exactly as you do in
   *scape4_Use.py* (with removable columns already filtered out).
4. **Merge** – concatenate both datasets and _back‑fill_ missing
   `organization` values in the API side using the scraped `org` column.
5. **Save** – write the final DataFrame to `fondue_dataset.csv` (UTF‑8‑SIG) and
   optionally the intermediate files too so your existing scripts still work.

The module also exposes a convenience helper `build_dataset()` so inside
`data‑en.ipynb` you can simply do:

```python
from fondue_pipeline import build_dataset
prepareDF = build_dataset(wsite_rows=1000, api_rows=20000)
# prepareDF is already cleaned – continue with your own analysis ↓
```
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ---------------------------------------------------------------------------
# 1)  ──────────────────────────────────────────────────────────  SCRAPER  ────
# ---------------------------------------------------------------------------

_FINISHED_BTN = "div.container-finish"
_CARD = "div.containerData"
_CARD_TIME = "div[style*='background: rgb(211, 246, 189)']"
_ORG_SPANS = "span.detailReportPost"
_TYPE = "div.div_tag_problemType_post"
_ADDR = "p.detailTimes.address"


def _parse_time(text: str) -> float | None:
    """Convert Thai time strings → hours (float)."""
    text = text.strip()
    m1 = re.search(r"เสร็จสิ้นใน\s+(\d+)\s*น", text)
    if m1:
        return int(m1.group(1)) / 60.0  # minutes → hours
    m2 = re.search(r"เสร็จสิ้นใน\s+(\d+):(\d+)\s*ชม", text)
    if m2:
        h, m = map(int, m2.groups())
        return (h * 60 + m) / 60.0
    return None


def _build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(), options=opts)


def scrape_fondue(n_rows: int = 1000, headless: bool = True) -> pd.DataFrame:
    """Scrape *n_rows* finished cards from the web UI and return a DataFrame."""
    driver = _build_driver(headless)
    wait = WebDriverWait(driver, 20)
    data: List[dict] = []

    try:
        driver.get("https://fondue.traffy.in.th/bangkok")
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, _FINISHED_BTN))).click()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, _CARD)))

        last_len, stalls = 0, 0
        while len(data) < n_rows and stalls < 10:
            cards = driver.find_elements(By.CSS_SELECTOR, _CARD)
            for card in cards[last_len:]:
                try:
                    time_div = card.find_element(By.CSS_SELECTOR, _CARD_TIME)
                    time_val = _parse_time(time_div.text)

                    spans = card.find_elements(By.CSS_SELECTOR, _ORG_SPANS)
                    org_val = spans[1].text.strip() if len(spans) > 1 else None

                    type_text = card.find_element(By.CSS_SELECTOR, _TYPE).text.strip()
                    type_list = [t.strip() for t in re.split(r"[&\n]", type_text) if t.strip()]

                    addr_text = card.find_element(By.CSS_SELECTOR, _ADDR).text.strip()
                    district_val = addr_text.split()[0] if addr_text else None

                    data.append(
                        {
                            "eta_hours": round(time_val, 3) if time_val is not None else np.nan,
                            "org": org_val,
                            "type": type_list,
                            "district": district_val,
                        }
                    )
                    if len(data) >= n_rows:
                        break
                except Exception:
                    # silently skip malformed cards
                    continue

            if len(data) == last_len:
                stalls += 1
            else:
                stalls = 0
                last_len = len(data)

            driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(2)

    finally:
        driver.quit()

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 2)  ─────────────────────────────────────────────────────────  CLEANING  ────
# ---------------------------------------------------------------------------

def _fix_type(obj):
    """Normalise the *type* column → `set[str]`."""
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return set()

    if isinstance(obj, (list, tuple, set)):
        raw = []
        for item in obj:
            raw.extend(re.split(r"[,\n]", str(item)))
    else:
        raw = re.split(r"[,\n]", str(obj))

    return {r.strip() for r in raw if r.strip()}


def _fix_org(text):
    # 1) fast‑exit for None / NaN scalars
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan

    # 2) join iterable containers
    if isinstance(text, (list, tuple, set)):
        text = ", ".join(map(str, text))

    # 3) final tidy‑up
    text = re.sub(r"^\s*โดย:\s*", "", str(text), flags=re.I)
    text = re.sub(r"\s+", ",", text.strip())

    return text or np.nan            # empty → NaN



def clean_fondue_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning rules as *cleanScape.py* to a DataFrame."""
    df = df.copy()

    # eta & log_eta already renamed in scraper; ensure both exist
    if "eta_hours" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "eta_hours"})

    df["log_eta"] = np.log1p(df["eta_hours"].astype(float))
    df["type"] = df["type"].apply(_fix_type)
    if "org" in df.columns:
        df["org"] = df["org"].apply(_fix_org)
    return df


# ---------------------------------------------------------------------------
# 3)  ─────────────────────────────────────────────────────────────  API  ────
# ---------------------------------------------------------------------------
_API_URL = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"
_DROP_FIELDS = {
    "org_action",
    "description",
    "message_id",
    "ticket_id",
    "photo_url",
    "after_photo",
    "address",
    "subdistrict",
    "province",
    "problem_type_abdul",
    "star",
    "count_reopen",
    "note",
    "description_reporter",
    "state_type_latest",
    "timestamp_inprogress",
    "timestamp_finished",
    "view_count",
    "problem_type_fondue",
    "see_info",
    "total_point",
    "like",
    "dislike",
    "duration_minutes_inprogress",
    "state",
    "duration_minutes_finished",
    "duration_minutes_total",
    "ai",
}


def _fetch_api_batch(limit: int, offset: int):
    params = {
        "output_format": "json",
        "state_type": "finish",
        "limit": limit,
        "offset": offset,
    }
    r = requests.get(_API_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("features", [])


def fetch_api_finished(max_rows: int = 20000, batch_size: int = 1000, start_offset: int = 0) -> pd.DataFrame:
    """Download finished cases from the official API and return a DataFrame."""
    records = []
    for offset in range(start_offset, start_offset + max_rows, batch_size):
        batch = _fetch_api_batch(batch_size, offset)
        if not batch:
            break
        records.extend(batch)

    if not records:
        return pd.DataFrame()

    all_keys = [k for k in records[0]["properties"].keys() if k not in _DROP_FIELDS]
    rows = [ {k: feat["properties"].get(k) for k in all_keys} for feat in records ]
    df = pd.DataFrame(rows)

    # Re‑use cleaning helpers where relevant
    if "type" in df.columns:
        df["type"] = df["type"].apply(_fix_type)
    if "org" in df.columns:
        df.rename(columns={"org": "organization"}, inplace=True)
        df["organization"] = df["organization"].apply(_fix_org)

    return df


# ---------------------------------------------------------------------------
# 4)  ─────────────────────────────────────────────────────────────  MERGE  ──
# ---------------------------------------------------------------------------

def merge_datasets(api_df: pd.DataFrame, scraped_df: pd.DataFrame) -> pd.DataFrame:
    """Return a single DataFrame with *organization* back‑filled from scrape."""
    if "organization" not in api_df.columns:
        api_df["organization"] = np.nan

    # align indices – simplest way is positionally combine_first()
    merged = api_df.copy()
    scraped_org = scraped_df.get("org")
    if scraped_org is not None:
        merged["organization"] = merged["organization"].combine_first(scraped_org)
    return merged


# ---------------------------------------------------------------------------
# 5)  ─────────────────────────────────────────────────────────────  API  ────
# ---------------------------------------------------------------------------

def build_dataset(
    wsite_rows: int = 1000,
    api_rows: int = 20000,
    headless: bool = True,
    save_intermediate: bool = False,
):
    """End‑to‑end helper – returns cleaned, merged DataFrame."""
    scraped_raw = scrape_fondue(wsite_rows, headless=headless)
    scraped_clean = clean_fondue_df(scraped_raw)

    api_df = fetch_api_finished(max_rows=api_rows)

    merged = merge_datasets(api_df, scraped_clean)

    if save_intermediate:
        scraped_clean.to_csv("fondue_scraped_clean.csv", index=False, encoding="utf-8-sig")
        api_df.to_csv("fondue_api_raw.csv", index=False, encoding="utf-8-sig")

    merged.to_csv("fondue_dataset.csv", index=False, encoding="utf-8-sig")
    return merged


# ---------------------------------------------------------------------------
# 6)  ─────────────────────────────────────────────────────────────  MAIN  ───
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Run full Fondue data pipeline → CSV")
    p.add_argument("-w", "--website", type=int, default=1000, help="rows to scrape from website (default 1000)")
    p.add_argument("-a", "--api", type=int, default=20000, help="rows to download from API (default 20000)")
    p.add_argument("--no-headless", action="store_false", dest="headless", help="show Chrome window while scraping")
    p.add_argument("--keep", action="store_true", dest="keep", help="save intermediate CSVs too")
    args = p.parse_args()

    df = build_dataset(args.website, args.api, headless=args.headless, save_intermediate=args.keep)
    print(f"✅ fondue_dataset.csv written with {len(df):,} rows and {len(df.columns)} columns")


if __name__ == "__main__":
    _cli()
