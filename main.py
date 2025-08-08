import os, math, json, time, datetime as dt
import logging
from dateutil import tz
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Dataset: SF Building Permits (`i98e-djp9`)
API_URL = os.environ.get("SF_API_URL", "https://data.sfgov.org/resource/i98e-djp9.json")
APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN")  # optional but recommended
PAGE_SIZE = 50000

SINCE = dt.datetime(2023, 1, 1, tzinfo=tz.gettz("America/Los_Angeles"))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("sf-permits")

HEADERS = {"Accept": "application/json"}
if APP_TOKEN:
    HEADERS["X-App-Token"] = APP_TOKEN

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def fetch_page(offset: int):
    # Add server-side filters if available (e.g., only rows with any residential use/status complete)
    since_str = f"{SINCE.date().isoformat()}T00:00:00.000"
    # Server-side where to match our Python-side filtering semantics without changing permit types
    where_clauses = [
        "issued_date IS NOT NULL",
        f"issued_date >= '{since_str}'",
        # new_units > 0 equivalent
        "coalesce(proposed_units::number, 0) - coalesce(existing_units::number, 0) > 0",
    ]
    params = {
        "$limit": PAGE_SIZE,
        "$offset": offset,
        # Example: server-side projection to reduce payload
        "$select": ",".join([
            "permit_number","permit_type","permit_type_definition",
            "filed_date","issued_date","completed_date","status","status_date",
            "block","lot","street_number","street_name","street_suffix",
            "proposed_use","existing_use","proposed_units","existing_units",
            "adu","supervisor_district","zipcode","record_id"
        ]),
        "$where": " AND ".join(where_clauses),
    }
    start = time.time()
    logger.debug("HTTP GET", extra={"url": API_URL, "params": params})
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
    elapsed_ms = int((time.time() - start) * 1000)
    if r.status_code >= 400:
        # Log body to help diagnose Socrata errors like unknown columns
        logger.error(
            "HTTP %s from API in %sms: %s",
            r.status_code, elapsed_ms, r.text[:2000],
        )
        r.raise_for_status()
    try:
        body = r.json()
    except Exception:
        logger.error("Non-JSON response in %sms: %s", elapsed_ms, r.text[:2000])
        raise
    # Socrata sometimes returns JSON error bodies with 200s (rare). Guard anyway.
    if isinstance(body, dict) and body.get("message"):
        logger.error("API error body: %s", json.dumps(body, indent=2)[:2000])
        raise RuntimeError(f"API error: {body.get('message')}")
    logger.info("Fetched page offset=%d rows=%d in %sms", offset, len(body), elapsed_ms)
    return body

def coerce_dates(df: pd.DataFrame):
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    # move to local tz for comparisons/display
    for col in df.columns:
        if "date" in col.lower():
            df[col] = df[col].dt.tz_convert("America/Los_Angeles")
    return df

def normalize_cols(df: pd.DataFrame):
    # Harmonize mixed-case from API vs notebook; keep canonical snake_case
    rename = {
        "Permit Number":"permit_number",
        "Permit Type":"permit_type",
        "Permit Type Definition":"permit_type_definition",
        "Filed Date":"filed_date",
        "Issued Date":"issued_date",
        "Completed Date":"completed_date",
        "Status":"status",
        "Status Date":"status_date",
        "Block":"block","Lot":"lot",
        "Street Number":"street_number","Street Name":"street_name","Street Suffix":"street_suffix",
        "Proposed Use":"proposed_use","Existing Use":"existing_use",
        "Proposed Units":"proposed_units","Existing Units":"existing_units",
        "Record ID":"record_id","Supervisor District":"supervisor_district",
        "Zipcode":"zipcode","ADU":"adu"
    }
    # lower-case everything then rename known columns back to canonical snake_case
    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    df = df.rename(columns=rename)
    # Explicitly fix any remaining known lowercase fields already in canonical form
    if "record_id" in df.columns:
        pass
    if "adu" in df.columns:
        pass
    return df

def compute_new_units(df: pd.DataFrame):
    df["proposed_units"] = pd.to_numeric(df["proposed_units"], errors="coerce")
    df["existing_units"] = pd.to_numeric(df["existing_units"], errors="coerce")
    df["new_units"] = df["proposed_units"].fillna(0) - df["existing_units"].fillna(0)
    df["apn"] = df["block"].astype(str) + "/" + df["lot"].astype(str)
    return df

def dedupe_reasonably(df: pd.DataFrame):
    """
    Avoid overcounting the same project by multiple follow-up permits.
    Heuristic: within (apn, street_number), keep the row with the
    max(new_units, then latest completed_date). This matches your ‘drop_duplicates’ intent but is stricter.
    """
    # Only rows with new_units > 0, then pick representative per site+street pairing
    eligible = df[df["new_units"] > 0].copy()
    eligible["sort_key"] = eligible["new_units"].fillna(0)
    eligible = eligible.sort_values(["apn","street_number","sort_key","completed_date"], ascending=[True, True, False, False])
    deduped = eligible.drop_duplicates(subset=["apn","street_number"], keep="first")
    return deduped

def main():
    # 1) Pull all pages
    all_rows = []
    offset = 0
    while True:
        logger.info("Requesting page offset=%d", offset)
        page = fetch_page(offset)
        if not page:
            break
        all_rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)  # be nice

    if not all_rows:
        logger.error("No data returned from API %s", API_URL)
        raise SystemExit("No data returned")

    df = pd.DataFrame(all_rows)
    logger.info("Raw dataframe shape=%s cols=%s", df.shape, list(df.columns))
    df = normalize_cols(df)
    logger.debug("Normalized cols=%s", list(df.columns))
    df = coerce_dates(df)
    df = compute_new_units(df)

    # 2) Filter to completed since 2023-01-01
    df_recent = df[(df["completed_date"].notna()) & (df["completed_date"] >= SINCE)]
    df_recent = df_recent[df_recent["new_units"] > 0]

    # Optional: focus on residential-ish uses (uncomment to narrow)
    # residential_uses = {
    #     "apartments","1 family dwelling","2 family dwelling",
    #     "residential hotel","misc group residns.","artist live/work",
    #     "convalescent home","accessory cottage","nursing home non amb",
    #     "r-3(dwg) nursing","nursing home gt 6","orphanage"
    # }
    # df_recent = df_recent[df_recent["proposed_use"].str.lower().isin({u.lower() for u in residential_uses})]

    # 3) Deduplicate to avoid counting follow-up OTC/fire-only/etc. permits
    deduped = dedupe_reasonably(df_recent)

    # 4) Aggregate
    total_units = int(deduped["new_units"].sum())

    # 5) Monthly series for chart
    m = deduped.copy()
    m["month"] = m["completed_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (m.groupby("month")["new_units"].sum()
                 .reset_index()
                 .sort_values("month"))
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m")

    # 6) Emit JSON artifacts
    now_pt = dt.datetime.now(tz=tz.gettz("America/Los_Angeles")).isoformat(timespec="seconds")
    out_totals = {
        "units_built_since_2023_01_01": total_units,
        "last_updated": now_pt,
        "methodology_version": 1,
    }
    out_monthly = monthly.to_dict(orient="records")

    os.makedirs("public_data", exist_ok=True)
    with open("public_data/totals.json","w") as f: json.dump(out_totals, f, indent=2)
    with open("public_data/monthly.json","w") as f: json.dump(out_monthly, f, indent=2)

    # 7) Keep today’s raw snapshot (optional, for audit)
    with open(f"public_data/raw_{now_pt[:10]}.json","w") as f: json.dump(all_rows, f)

    logger.info("OK total=%d rows=%d deduped=%d", total_units, len(df), len(deduped))
    logger.debug("Sample rows:\n%s", deduped.head(5).to_string())

if __name__ == "__main__":
    main()
