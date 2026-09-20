import datetime
import logging

from strands import tool

from tools.mandi import agmarknet, store

logger = logging.getLogger(__name__)

MAX_MARKETS = 40


def _num(value):
    """Keep whole rupee figures as ints so the payload reads 5950, not 5950.0."""
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    return None


def _result(rows: list[dict], commodity: str, state: str, report_date: str) -> dict:
    rows = sorted(rows, key=lambda row: (row.get("market", ""), row.get("variety", "")))
    names = sorted({row["commodity"] for row in rows if row.get("commodity")})
    price_unit = next((row.get("price_unit") for row in rows if row.get("price_unit")), "Rs./Quintal")
    included = rows[:MAX_MARKETS]

    result = {
        "commodity": ", ".join(names) or commodity,
        "state": state.title(),
        "source": "AgMarkNet government market price service",
        "source_url": "https://agmarknet.gov.in/",
        "report_date": report_date,
        "is_todays_report": report_date == datetime.date.today().isoformat(),
        "price_unit": price_unit,
        "markets_reporting": len({row.get("market") for row in rows}),
        "prices": [
            {
                "market": row.get("market"),
                "variety": row.get("variety") or None,
                "min_price": _num(row.get("min_price")),
                "max_price": _num(row.get("max_price")),
                "modal_price": _num(row.get("modal_price")),
            }
            for row in included
        ],
    }

    modals = [row["modal_price"] for row in rows if isinstance(row.get("modal_price"), (int, float))]
    if modals:
        result["modal_price_summary"] = {
            "low": _num(min(modals)),
            "high": _num(max(modals)),
            "average": _num(round(sum(modals) / len(modals))),
        }

    if not result["is_todays_report"]:
        result["freshness_note"] = "Today's report is not published yet; this is the most recent one."
    if len(rows) > len(included):
        result["truncation_note"] = (
            f"Listing {len(included)} of {len(rows)} entries; modal_price_summary covers all of them."
        )
    return result


@tool
def mandi_price(commodity: str, state: str, market: str = "") -> dict:
    """Get mandi (market) prices for a crop from AgMarkNet, the government market price service.

    Args:
        commodity: Crop or commodity name, e.g. "onion", "paddy", "tomato".
        state: The state the farmer is in, e.g. "Tamil Nadu". Required — prices are published per state.
        market: Optional mandi name to narrow the result to one market.
    """
    if not commodity.strip():
        return {"error": "Specify which crop to price, e.g. commodity='onion', state='Tamil Nadu'."}
    if not state.strip():
        return {"error": "Specify the farmer's state — AgMarkNet publishes prices per state."}

    today = datetime.date.today().isoformat()
    stored_rows, stored_date, fetched_on = store.read_latest(commodity, state)
    stored_state = stored_rows[0].get("state", state) if stored_rows else state

    # Serve from the table when it already holds today's report, or when today's scrape
    # already ran and upstream had nothing newer (reports lag by a day or more).
    if stored_rows and (stored_date == today or fetched_on == today):
        rows, report_date, state_name = stored_rows, stored_date, stored_state
    else:
        try:
            scraped, state_name = agmarknet.fetch_prices(commodity, state)
        except agmarknet.AgMarkNetError as e:
            logger.warning("mandi price fetch failed: %s", e)
            if stored_rows:
                return _result(stored_rows, commodity, stored_state, stored_date)
            return {"error": f"Could not fetch mandi prices for {commodity} in {state}: {e}"}

        if scraped:
            store.write_prices(commodity, state, scraped, today)
            rows, report_date = scraped, scraped[0]["report_date"]
        elif stored_rows:
            rows, report_date, state_name = stored_rows, stored_date, stored_state
        else:
            return {
                "commodity": commodity,
                "state": state_name.title(),
                "prices": [],
                "message": (f"AgMarkNet published no prices for '{commodity}' in {state_name.title()} "
                            f"in the last {agmarknet.LOOKBACK_DAYS} days."),
            }

    if market.strip():
        wanted = market.strip().lower()
        narrowed = [row for row in rows if wanted in row.get("market", "").lower()]
        if not narrowed:
            return {
                "error": f"No entries for market '{market}'.",
                "markets_available": sorted({row.get("market", "") for row in rows})[:25],
                "report_date": report_date,
            }
        rows = narrowed

    return _result(rows, commodity, state_name, report_date)
