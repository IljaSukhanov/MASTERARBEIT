from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
JHU_PATH = HERE / "time_series_covid19_confirmed_US.csv"
ORDERS_PATH = HERE / "littler_statewide_orders.html"
COUNTY_OUTPUT = HERE / "county_interval_lengths.csv"
STATE_OUTPUT = HERE / "state_order_end_dates.csv"
SUMMARY_OUTPUT = HERE / "reconstruction_summary.txt"
FIGURE_PNG = PROJECT / "Abschlussarbeit_2021/LaTeX/images/covid_Li_histogram.png"
FIGURE_PDF = PROJECT / "Abschlussarbeit_2021/LaTeX/images/covid_Li_histogram.pdf"

US_STATES_AND_DC = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
}

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
DATE_PATTERN = re.compile(rf"({MONTHS})\s+\d{{1,2}}(?:,\s*2020)?", re.I)


def normalize_state(value: str) -> str:
    state = re.split(r"[\u2013\u2014]", value, maxsplit=1)[0]
    state = re.sub(r"[\*\d]+$", "", state).strip()
    return "Washington" if state == "Washington State" else state


def parse_order_table(path: Path) -> pd.DataFrame:
    html = path.read_text(encoding="utf-8")
    table = pd.read_html(StringIO(html))[0]
    table.columns = ["state_raw", "effective_date_raw", "end_date_raw", "resources"]
    table = table.iloc[1:].copy()
    table["state"] = table["state_raw"].map(normalize_state)

    def extract_end_date(text: str) -> tuple[pd.Timestamp | pd.NaT, str]:
        text = str(text)
        match = DATE_PATTERN.search(text)
        if not match:
            return pd.NaT, "no dated statewide end in source"

        date_text = match.group(0)
        if "2020" not in date_text:
            date_text += ", 2020"
        end_date = pd.to_datetime(date_text).normalize()
        prefix = text[: match.start()].lower()
        if "phased" in prefix or "reopening" in prefix:
            basis = "phased reopening date used as proxy"
        else:
            basis = "listed duration/end date"
        return end_date, basis

    parsed = table["end_date_raw"].map(extract_end_date)
    table[["restriction_end_date", "date_basis"]] = pd.DataFrame(
        parsed.tolist(), index=table.index
    )
    return table[
        [
            "state",
            "restriction_end_date",
            "date_basis",
            "effective_date_raw",
            "end_date_raw",
        ]
    ].sort_values("state")


def load_counties(path: Path) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(path, low_memory=False)
    date_columns = [column for column in data.columns if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", column)]

    counties = data[
        data["iso2"].eq("US")
        & data["FIPS"].notna()
        & data["Province_State"].isin(US_STATES_AND_DC)
        & ~data["Admin2"].fillna("").str.contains("Unassigned|Out of", case=False, regex=True)
    ].copy()

    # The final JHU file contains both the former Valdez-Cordova Census Area and
    # its two successors. The 2020 study's 3,142-county universe predates the split.
    counties = counties[~counties["FIPS"].isin([2063.0, 2066.0])].copy()
    if len(counties) != 3142:
        raise RuntimeError(f"Expected 3,142 county equivalents, found {len(counties):,}.")
    return counties, date_columns


def first_second_case_date(row: pd.Series, date_columns: list[str]) -> pd.Timestamp | pd.NaT:
    values = pd.to_numeric(row[date_columns], errors="coerce").fillna(0).to_numpy()
    hits = np.flatnonzero(values >= 2)
    if not len(hits):
        return pd.NaT
    return pd.to_datetime(date_columns[int(hits[0])], format="%m/%d/%y")


def build_intervals(counties: pd.DataFrame, date_columns: list[str], orders: pd.DataFrame) -> pd.DataFrame:
    result = counties[["FIPS", "Admin2", "Province_State", "Combined_Key"]].copy()
    result.columns = ["fips", "county", "state", "combined_key"]
    result["second_case_date"] = counties.apply(
        first_second_case_date, axis=1, date_columns=date_columns
    ).to_numpy()

    result = result.merge(
        orders[["state", "restriction_end_date", "date_basis"]],
        on="state",
        how="left",
        validate="many_to_one",
    )
    result["L_i"] = (
        result["restriction_end_date"] - result["second_case_date"]
    ).dt.days + 1

    result["status"] = "included"
    result.loc[result["second_case_date"].isna(), "status"] = "no recorded second case"
    result.loc[result["restriction_end_date"].isna(), "status"] = "no dated statewide end in source"
    result.loc[result["L_i"].le(0), "status"] = "second case after statewide end"
    return result.sort_values(["state", "county"])


def make_histogram(intervals: pd.DataFrame) -> None:
    included = intervals[intervals["status"].eq("included")].copy()
    exact = included[included["date_basis"].eq("listed duration/end date")]["L_i"]
    proxy = included[included["date_basis"].eq("phased reopening date used as proxy")]["L_i"]
    values = included["L_i"]

    upper = int(np.ceil(values.max() / 5) * 5)
    bins = np.arange(0.5, upper + 5.5, 5)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#C9D4D8",
            "axes.labelcolor": "#333333",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7FAFB")
    ax.hist(
        [exact, proxy],
        bins=bins,
        stacked=True,
        color=["#00687D", "#85C7D2"],
        edgecolor="white",
        linewidth=0.7,
        label=["Listed statewide end date", "Phased-reopening proxy"],
    )
    ax.axvline(39, color="#D06B35", linewidth=2.2, linestyle="--", label="39-day target")
    ax.axvline(values.median(), color="#555555", linewidth=1.5, linestyle=":", label=f"Median: {values.median():.0f} days")
    ax.set_title("Raw county observation spans before standardization", loc="left", fontsize=18, pad=16)
    ax.set_xlabel(r"$L_i=r_i-s_i+1$ (days)")
    ax.set_ylabel("Number of counties")
    ax.grid(axis="y", color="#DDE5E8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper right")

    q1, q3 = values.quantile([0.25, 0.75])
    subtitle = (
        f"Included n={len(values):,}; median={values.median():.0f} days; "
        f"IQR={q1:.0f}-{q3:.0f} days"
    )
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, color="#555555", va="bottom")
    excluded = len(intervals) - len(included)
    fig.text(
        0.10,
        0.02,
        f"Reconstruction from JHU CSSE county cases and the cited Littler statewide-order table. "
        f"Excluded: {excluded:,} counties without a usable positive span.",
        fontsize=9,
        color="#666666",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.15)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=300, facecolor="white")
    fig.savefig(FIGURE_PDF, facecolor="white")
    plt.close(fig)


def write_summary(intervals: pd.DataFrame, orders: pd.DataFrame) -> None:
    included = intervals[intervals["status"].eq("included")]
    lines = [
        "COVID interval-length reconstruction",
        "=====================================",
        f"County equivalents in reconstruction: {len(intervals):,}",
        f"Included positive L_i values: {len(included):,}",
        f"Median L_i: {included['L_i'].median():.1f}",
        f"Mean L_i: {included['L_i'].mean():.1f}",
        f"IQR: {included['L_i'].quantile(0.25):.1f} to {included['L_i'].quantile(0.75):.1f}",
        f"Range: {included['L_i'].min():.0f} to {included['L_i'].max():.0f}",
        "",
        "Status counts:",
        intervals["status"].value_counts().to_string(),
        "",
        "States absent from the cited order table:",
        ", ".join(sorted(US_STATES_AND_DC - set(orders["state"]))),
        "",
        "States listed without a dated statewide end:",
        ", ".join(orders.loc[orders["restriction_end_date"].isna(), "state"]),
    ]
    SUMMARY_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    orders = parse_order_table(ORDERS_PATH)
    counties, date_columns = load_counties(JHU_PATH)
    intervals = build_intervals(counties, date_columns, orders)

    orders.to_csv(STATE_OUTPUT, index=False, date_format="%Y-%m-%d")
    intervals.to_csv(COUNTY_OUTPUT, index=False, date_format="%Y-%m-%d")
    write_summary(intervals, orders)
    make_histogram(intervals)
    print(SUMMARY_OUTPUT.read_text(encoding="utf-8"))
    print(f"Wrote {FIGURE_PNG}")


if __name__ == "__main__":
    main()
