import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from curl_cffi import requests


OUTPUT_FILE = Path(__file__).with_name("raw_historical_indices.xlsx")
API_URL = "https://api.investing.com/api/financialdata/historical/{}"
PAGE_URL = "https://vn.investing.com/indices/{}-historical-data"
TECHNICAL_URL = "https://vn.investing.com/indices/{}-technical"
MIN_DATE = date(2026, 8, 1)
CHUNK_DAYS = 15 * 365
LOOKBACK_DAYS = 14
REQUEST_DELAY_SECONDS = 1
OUTPUT_COLUMNS = [
    "trading_date",
    "index_name",
    "close_price",
    "price_change",
    "end_of_prev_year_price",
    "MA20",
    "RSI",
    "ytd",
    "end_of_prev_year_date",
    "etl_date",
]
INDICES = {
    "VN Index": ("41063", "vn", "vn_index"),
    "HNX Index": ("41062", "hnx", "hnx_index"),
    "S&P 500": ("166", "us-spx-500", "sp500"),
    "Dow Jones": ("169", "us-30", "dow_jones"),
    "Shanghai Composite": ("40820", "shanghai-composite", "shanghai_composite"),
    "SZSE Component": ("942630", "szse-component", "szse_component"),
    "Hang Seng": ("179", "hang-sen-40", "hang_seng"),
    "Nikkei 225": ("178", "japan-ni225", "nikkei_225"),
    "KOSPI": ("37426", "kospi", "kospi"),
    "VN 30": ("41064", "vn-30", "vn30"),
}


def parse_date(value):
    text = str(value).strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    for pattern in ("%d/%m/%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    raise RuntimeError(f"Unsupported date: {value!r}")


def numeric_column(data, *names):
    values = pd.Series(float("nan"), index=data.index)
    for name in names:
        if name in data:
            column = pd.to_numeric(
                data[name].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            values = values.fillna(column)
    return values


def calculate_ytd(data, baselines):
    data = data.copy()
    data["trading_date"] = pd.to_datetime(data["trading_date"], errors="raise")
    data["etl_date"] = pd.to_datetime(data["etl_date"], errors="coerce")
    data["close_price"] = pd.to_numeric(data["close_price"], errors="coerce")
    data["price_change"] = pd.to_numeric(data["price_change"], errors="coerce")
    data["target_year"] = data["trading_date"].dt.year
    data = data.drop(
        columns=["end_of_prev_year_price", "ytd", "end_of_prev_year_date"],
        errors="ignore",
    ).merge(baselines, on=["index_name", "target_year"], how="left")
    data["end_of_prev_year_date"] = pd.to_datetime(
        data["end_of_prev_year_date"], errors="coerce"
    )
    data["ytd"] = (
        data["close_price"] / data["end_of_prev_year_price"] - 1
    ) * 100
    missing_close = data["close_price"].isna()
    data.loc[missing_close, "ytd"] = pd.NA
    order = {details[2]: position for position, details in enumerate(INDICES.values())}
    data["_index_order"] = data["index_name"].map(order)
    data = data.sort_values(
        ["_index_order", "trading_date"], ascending=[True, False], kind="stable"
    )
    return data.reindex(columns=OUTPUT_COLUMNS).reset_index(drop=True)


def load_existing_data():
    if not OUTPUT_FILE.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = pd.read_excel(OUTPUT_FILE, sheet_name="Historical", keep_default_na=False)
    missing = set(OUTPUT_COLUMNS) - set(data.columns)
    if missing - {"MA20", "RSI"}:
        raise RuntimeError("Existing Excel file does not match the output layout")
    for column in missing:
        data[column] = ""
    data = data[OUTPUT_COLUMNS].drop_duplicates(["trading_date", "index_name"], keep="last")
    data["trading_date"] = pd.to_datetime(data["trading_date"], errors="raise")
    data["etl_date"] = pd.to_datetime(data["etl_date"], errors="coerce")
    return data


def check_output_file():
    if OUTPUT_FILE.exists():
        try:
            with OUTPUT_FILE.open("r+b"):
                pass
        except PermissionError as error:
            raise RuntimeError(f"Close {OUTPUT_FILE.name} in Excel, then run again") from error


def get(session, url, **kwargs):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=60, **kwargs)
            response.raise_for_status()
            return response
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def fetch_range(session, instrument_id, source_url, start, end):
    response = get(
        session,
        API_URL.format(instrument_id),
        params={
            "start-date": start.isoformat(),
            "end-date": end.isoformat(),
            "time-frame": "Daily",
            "add-missing-rows": "false",
        },
        headers={"domain-id": "vn", "Referer": source_url},
    )
    rows = response.json().get("data") or []
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid historical response for {source_url}")
    if len(rows) >= 4999:
        if start >= end:
            raise RuntimeError(f"Investing truncated data for {source_url}")
        middle = start + (end - start) // 2
        first = fetch_range(session, instrument_id, source_url, start, middle)
        time.sleep(REQUEST_DELAY_SECONDS)
        return first + fetch_range(
            session, instrument_id, source_url, middle + timedelta(days=1), end
        )
    return rows


def fetch_technical(session, slug):
    response = get(
        session,
        TECHNICAL_URL.format(slug),
        headers={"Accept": "text/html", "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8"},
    )
    prefix = '"technicalStore":{"technicalData":'
    position = response.text.find(prefix)
    if position < 0:
        raise RuntimeError(f"Technical data not found for {slug}")
    technical, _ = json.JSONDecoder().raw_decode(
        response.text[position + len(prefix):]
    )
    return (
        float(technical["movingAverages"]["simple"]["SMA20"]),
        float(technical["indicators"]["rsi"]["value"]),
    )


def fetch_year_end(session, instrument_id, slug, year):
    source_url = PAGE_URL.format(slug)
    end = date(year - 1, 12, 31)
    rows = fetch_range(session, instrument_id, source_url, end - timedelta(days=14), end)
    if not rows:
        raise RuntimeError(f"Previous-year close not found for {slug} in {year - 1}")
    row = max(rows, key=lambda item: parse_date(item.get("rowDateTimestamp") or item.get("rowDate")))
    price = row.get("last_closeRaw") or row.get("last_close")
    return float(str(price).replace(",", "")), parse_date(
        row.get("rowDateTimestamp") or row.get("rowDate")
    )


def crawl_index(session, index_name, instrument_id, slug, existing, etl_date):
    source_url = PAGE_URL.format(slug)
    today = datetime.now().date()
    old_dates = (
        existing.loc[existing["index_name"].eq(index_name), "trading_date"]
        if not existing.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    batches = []

    def collect(start, end):
        rows = fetch_range(session, instrument_id, source_url, start, end)
        batches.extend(
            {
                "trading_date": parse_date(
                    row.get("rowDateTimestamp") or row.get("rowDate")
                ),
                "index_name": index_name,
                "close_price": row.get("last_closeRaw") or row.get("last_close"),
                "price_change": row.get("change_precentRaw") or row.get("change_precent"),
                "etl_date": etl_date,
            }
            for row in rows
        )
        print(f"    {start} -> {end}: {len(rows)} rows")
        time.sleep(REQUEST_DELAY_SECONDS)
        return rows

    if not old_dates.empty:
        start = (
            MIN_DATE
            if old_dates.min().date() > MIN_DATE
            else max(MIN_DATE, old_dates.max().date() - timedelta(days=LOOKBACK_DAYS))
        )
        collect(start, today)
        return batches

    end = today
    while end >= MIN_DATE:
        start = max(MIN_DATE, end - timedelta(days=CHUNK_DAYS))
        rows = collect(start, end)
        if not rows:
            break
        oldest = min(
            date.fromisoformat(
                parse_date(row.get("rowDateTimestamp") or row.get("rowDate"))
            )
            for row in rows
        )
        if oldest > start + timedelta(days=31):
            break
        end = start - timedelta(days=1)
    return batches


def crawl_historical_indices(existing, etl_date):
    session = requests.Session(
        impersonate="chrome",
        headers={
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        },
    )
    records = []
    technical = {}
    try:
        for position, (name, (instrument_id, slug, index_name)) in enumerate(
            INDICES.items(), start=1
        ):
            print(f"[{position}/{len(INDICES)}] Crawling {index_name}...")
            records.extend(
                crawl_index(
                    session, index_name, instrument_id, slug, existing, etl_date
                )
            )
            technical[index_name] = fetch_technical(session, slug)
        if {row["index_name"] for row in records} != {
            details[2] for details in INDICES.values()
        }:
            raise RuntimeError("One or more indices returned no data")
        fresh = pd.DataFrame(records)
        fresh["trading_date"] = pd.to_datetime(fresh["trading_date"], errors="raise")
        fresh["close_price"] = numeric_column(fresh, "close_price")
        fresh["price_change"] = numeric_column(fresh, "price_change") / 100
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = combined.drop_duplicates(["trading_date", "index_name"], keep="last")
        baseline_rows = []
        for _, (instrument_id, slug, index_name) in INDICES.items():
            years = sorted(
                combined.loc[
                    combined["index_name"].eq(index_name), "trading_date"
                ].dt.year.unique()
            )
            for year in years:
                price, baseline_date = fetch_year_end(
                    session, instrument_id, slug, int(year)
                )
                baseline_rows.append(
                    {
                        "index_name": index_name,
                        "target_year": int(year),
                        "end_of_prev_year_price": price,
                        "end_of_prev_year_date": baseline_date,
                    }
                )
                time.sleep(REQUEST_DELAY_SECONDS)
    finally:
        session.close()
    return combined, pd.DataFrame(baseline_rows), technical


def add_future_rows(data, etl_date):
    rows = []
    today = datetime.now().date()
    for index_name in [details[2] for details in INDICES.values()]:
        current = data.loc[data["index_name"].eq(index_name), "trading_date"]
        if current.empty:
            continue
        start = current.max().date() + timedelta(days=1)
        rows.extend(
            {
                "trading_date": trading_date,
                "index_name": index_name,
                "etl_date": etl_date,
            }
            for trading_date in pd.bdate_range(start, today).date
        )
    if not rows:
        return data
    return pd.concat([data, pd.DataFrame(rows)], ignore_index=True)


def save_excel(data):
    temporary = OUTPUT_FILE.with_name(f".{OUTPUT_FILE.stem}.tmp.xlsx")
    output = data.copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.date
    try:
        with pd.ExcelWriter(
            temporary,
            engine="openpyxl",
            date_format="yyyy-mm-dd",
            datetime_format="yyyy-mm-dd hh:mm:ss",
        ) as writer:
            output.to_excel(
                writer, index=False, sheet_name="Historical", columns=OUTPUT_COLUMNS
            )
        temporary.replace(OUTPUT_FILE)
    except PermissionError as error:
        raise RuntimeError(f"Close {OUTPUT_FILE.name} in Excel, then run again") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    try:
        check_output_file()
        existing = load_existing_data()
        existing = existing.loc[existing["trading_date"].dt.date >= MIN_DATE]
        etl_date = datetime.now().replace(microsecond=0)
        combined, baselines, technical = crawl_historical_indices(existing, etl_date)
        combined = add_future_rows(combined, etl_date)
        output = calculate_ytd(combined, baselines)
        output[["MA20", "RSI"]] = pd.NA
        for index_name, (ma20, rsi) in technical.items():
            index_mask = output["index_name"].eq(index_name)
            latest_price_date = output.loc[
                index_mask & output["close_price"].notna(), "trading_date"
            ].max()
            if pd.notna(latest_price_date):
                output.loc[
                    index_mask & output["trading_date"].ge(latest_price_date),
                    ["MA20", "RSI"],
                ] = [ma20, rsi]
        save_excel(output)
        print(f"Saved {len(combined)} rows / {OUTPUT_FILE}")
    except Exception as error:
        print(f"Historical crawl failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
