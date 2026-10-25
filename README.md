# Capstone 1: Getting into Business
## Real Estate Investment Data Exploration
**Author: Zetong Li**  

## Part 1: Understanding the Data
### FHFA State House Price Index Data

This report explores the quarterly FHFA All-Transactions House Price Index (HPI) for states. Each row describes one geographic area in one quarter. The index tracks changes in single-family home values. It does not report the sale price of an individual home.

### When was the data acquired?

The data cover **1975 Q1 through 2026 Q2**. For this report, the original CSV was downloaded from FHFA on **October 25, 2026**. This download date is different from the dates of the housing transactions. The CSV does not include the collection date of each transaction. FHFA may revise earlier index values as new information becomes available.

### Where was the data acquired?

The file was obtained directly from the [FHFA House Price Index data page](https://www.fhfa.gov/data/house-price-index). The selected dataset covers **all 50 U.S. states and Washington, DC**. It does not contain individual addresses or neighborhood-level observations.

### How was the data acquired?

FHFA uses mortgage information from Fannie Mae and Freddie Mac. This All-Transactions index includes home sale prices and refinance appraisal values. FHFA applies a weighted repeat-sales method to compare values for the same properties over time. The published CSV contains the resulting indexes rather than the original mortgage records. More information is available in the [FHFA HPI FAQs](https://www.fhfa.gov/faqs/hpi).

### What are the attributes of this dataset?

The original CSV contains **four columns and 10,506 rows**. It has no header, so the code assigns the column names below.

**Table 1. Data Dictionary**

| Code | Attribute | Definition | Data type | Measurement scale |
|---|---|---|---|---|
| 1 | state | Two-letter abbreviation for a state or Washington, DC | Categorical | Nominal |
| 2 | year | Calendar year of the observation | Discrete numerical time variable | Interval |
| 3 | quarter | Quarter of the year, coded 1, 2, 3, or 4 | Ordered time category | Ordinal within a year; cyclic across years |
| 4 | hpi | All-Transactions house price index, not seasonally adjusted, with 1980 Q1 = 100 | Continuous numerical index | Ratio within a consistent area and index series |

### What type of data do these attributes contain?

**Nominal:** `state` identifies locations. State names have no numerical order.

**Interval:** `year` records calendar time. Differences between years are meaningful, but calendar year zero does not represent an absence of time.

**Ordinal:** `quarter` orders observations within a year. After Q4, the sequence returns to Q1 of the next year, so the year must also be used when ordering observations.

**Ratio:** `hpi` describes a relative price level within the same index series. An increase from 100 to 200 represents a doubling of that area's estimated price level. The index is not measured in dollars. Comparing index values between states does not reveal which state has more expensive homes.

### Load the State Time Series Dataset

The code uses the unchanged FHFA snapshot included in the repository. If that snapshot is unavailable, such as when opening only the notebook in Colab, it downloads the CSV automatically from FHFA. No manual dataset upload is needed. A live download can contain revised values or additional quarters.


```python
from pathlib import Path
from io import BytesIO
from urllib.request import Request, urlopen
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOURCE_URL = "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_state.csv"
SNAPSHOT = Path("data/hpi_at_state.csv")
EXPECTED_SHA256 = "b6bf3687c230c2ad008657fdb0c2b7489b5f99b6d1193f72c46c68d88df83d90"
USE_SNAPSHOT = True
if USE_SNAPSHOT and SNAPSHOT.exists():
    payload = SNAPSHOT.read_bytes()
    origin = "repository snapshot acquired 2026-10-25"
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256, "Snapshot changed"
else:
    with urlopen(Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
        payload = response.read()
    origin = "FHFA live download on " + datetime.now(ZoneInfo("America/New_York")).isoformat()
print("Source:", origin)
print("SHA-256:", hashlib.sha256(payload).hexdigest())
# The CSV has no header: header=None preserves the first observation.
raw = pd.read_csv(BytesIO(payload), header=None,
                  names=["state", "year", "quarter", "hpi"],
                  na_values=["", " ", ".", "NA", "N/A", "null"])
print("Rows and columns:", raw.shape)
print(raw.head().to_string(index=False))
print("\nStorage types:\n" + raw.dtypes.to_string())
```

```text
Source: repository snapshot acquired 2026-10-25
SHA-256: b6bf3687c230c2ad008657fdb0c2b7489b5f99b6d1193f72c46c68d88df83d90
Rows and columns: (10506, 4)
state  year  quarter   hpi
   AK  1975        1 69.15
   AK  1975        2 69.06
   AK  1975        3 66.90
   AK  1975        4 67.58
   AK  1976        1 70.63

Storage types:
state          str
year         int64
quarter      int64
hpi        float64
```

The first rows above show the structure of the dataset. `state` is stored as text, `year` and `quarter` as integers, and `hpi` as a decimal number. Storage type and measurement scale are different. For example, a quarter is stored as an integer but represents an ordered time category.

## Part 2: Data Summary and Initial Insights
### Missing Values and Data Quality


```python
df = raw.copy()
df["state"] = df["state"].astype("string").str.strip().replace("", pd.NA)
for col in ["year", "quarter", "hpi"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
quality = pd.DataFrame({"missing_count": df.isna().sum(),
                        "missing_pct": df.isna().mean() * 100})
print(quality.to_string())
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate state-quarter keys:", df.duplicated(["state", "year", "quarter"]).sum())
assert not df[["state", "year", "quarter"]].isna().any().any()
assert df["quarter"].isin([1, 2, 3, 4]).all()
assert (df["year"] % 1 == 0).all()
assert (df["hpi"].dropna() > 0).all()
assert not df.duplicated(["state", "year", "quarter"]).any()
df[["year", "quarter"]] = df[["year", "quarter"]].astype(int)
df["period"] = pd.PeriodIndex(df["year"].astype(str) + "Q" + df["quarter"].astype(str), freq="Q")
df = df.sort_values(["state", "period"]).reset_index(drop=True)
wide = df.pivot(index="period", columns="state", values="hpi")
full_periods = pd.period_range(wide.index.min(), wide.index.max(), freq="Q")
wide = wide.reindex(full_periods)
print("Time range:", wide.index.min(), "to", wide.index.max())
print("Geographic areas:", len(wide.columns))
print("Areas:", ", ".join(wide.columns))
print("Quarters:", len(wide))
print("Missing state-quarter cells, including absent rows:", int(wide.isna().sum().sum()))
print("Rows per area (minimum, maximum):", df.groupby("state").size().min(), df.groupby("state").size().max())
```

```text
         missing_count  missing_pct
state                0          0.0
year                 0          0.0
quarter              0          0.0
hpi                  0          0.0
Exact duplicate rows: 0
Duplicate state-quarter keys: 0
Time range: 1975Q1 to 2026Q2
Geographic areas: 51
Areas: AK, AL, AR, AZ, CA, CO, CT, DC, DE, FL, GA, HI, IA, ID, IL, IN, KS, KY, LA, MA, MD, ME, MI, MN, MO, MS, MT, NC, ND, NE, NH, NJ, NM, NV, NY, OH, OK, OR, PA, RI, SC, SD, TN, TX, UT, VA, VT, WA, WI, WV, WY
Quarters: 206
Missing state-quarter cells, including absent rows: 0
Rows per area (minimum, maximum): 206 206
```

In the saved snapshot, all four source columns have **zero missing values**, there are **zero duplicate rows or duplicate state-quarter keys**, and the panel has **zero absent state-quarter cells**. Each of the 51 areas has 206 quarters. No observations were deleted or imputed.

If a future file contains missing HPI values, first check the original source and document the affected periods. Leave unavailable observations as missing for descriptive analysis and report valid counts. Do not replace missing prices or indexes with zero. Interpolation could be considered only for short internal gaps with a stated reason, and should be flagged. Missing state or time identifiers require investigation before calculating growth. Extreme values should be checked, not automatically removed.

### Summary Statistics for the State Time Series Dataset


```python
# Create a formatted Markdown table without extra table packages.
def show_table(frame, caption):
    print(caption)
    print(frame.to_string(float_format=lambda value: f"{value:,.2f}"))

# Summary statistics: numerical columns only.
# Year and quarter summaries describe coverage, not housing performance.
numeric = df[["year", "quarter", "hpi"]]
summary = numeric.agg(["mean", "std", "median", "min", "max"]).rename(
    index={"mean": "Mean", "std": "Stdev", "median": "Median",
           "min": "Minimum", "max": "Maximum"})
summary.loc["Range"] = numeric.max() - numeric.min()
summary.loc["NAs"] = numeric.isna().sum()
show_table(summary, "Table 2. Summary Statistics")
print("\nHPI mode:", df["hpi"].mode().tolist())
print("All state codes are tied as modes; each occurs 206 times in the snapshot.")

# Calculate annual percentage changes, using a complete quarterly calendar.
yoy = wide.pct_change(periods=4, fill_method=None) * 100
latest_period = wide.index.max()
latest = pd.DataFrame({"hpi": wide.loc[latest_period],
                       "yoy_pct": yoy.loc[latest_period]})
show_table(latest.describe().T, "\nLatest-quarter statistics: " + str(latest_period))
show_table(latest.nlargest(5, "yoy_pct"), "\nFive highest annual growth rates")
show_table(latest.nsmallest(5, "yoy_pct"), "\nFive lowest annual growth rates")
print("\nAreas with positive annual growth:", int((latest.yoy_pct > 0).sum()))
print("Pennsylvania annual growth (%):", round(latest.loc["PA", "yoy_pct"], 2))
print("Expected missing annual-growth observations:", int(yoy.isna().sum().sum()))
```

```text
Table 2. Summary Statistics
            year  quarter      hpi
Mean    2,000.25     2.49   274.89
Stdev      14.87     1.12   190.87
Median  2,000.00     2.00   229.25
Minimum 1,975.00     1.00    41.65
Maximum 2,026.00     4.00 1,326.94
Range      51.00     3.00 1,285.29
NAs         0.00     0.00     0.00

HPI mode: [100.0]
All state codes are tied as modes; each occurs 206 times in the snapshot.

Latest-quarter statistics: 2026Q2
         count   mean    std    min    25%    50%    75%      max
hpi      51.00 716.00 219.63 365.44 544.01 705.37 860.16 1,326.94
yoy_pct  51.00   3.44   1.75  -0.59   2.42   3.24   4.44     8.70

Five highest annual growth rates
         hpi  yoy_pct
state                
HI    982.57     8.70
WV    365.44     8.02
IL    565.87     5.99
CT    736.50     5.67
WI    646.66     5.60

Five lowest annual growth rates
           hpi  yoy_pct
state                  
DC    1,037.72    -0.59
CO      853.19     0.20
CA      974.97     0.90
AZ      731.21     0.93
WA    1,033.33     1.05

Areas with positive annual growth: 50
Pennsylvania annual growth (%): 4.65
Expected missing annual-growth observations: 204
```

**Table 2. Summary Statistics (Rendered)**

| Statistic | year | quarter | hpi |
| --- | --- | --- | --- |
| Mean | 2,000.25 | 2.49 | 274.89 |
| Stdev | 14.87 | 1.12 | 190.87 |
| Median | 2,000.00 | 2.00 | 229.25 |
| Minimum | 1,975.00 | 1.00 | 41.65 |
| Maximum | 2,026.00 | 4.00 | 1,326.94 |
| Range | 51.00 | 3.00 | 1,285.29 |
| NAs | 0.00 | 0.00 | 0.00 |

Stdev is the sample standard deviation. NAs counts missing values. The nominal state variable is summarized by frequency rather than a numerical mean.


### Findings and Observations

The dataset contains **10,506 observations** from **1975Q1 to 2026Q2**. Across all state-quarter observations, HPI has a mean of **274.89**, median of **229.25**, and sample standard deviation of **190.87**. The minimum is **41.65**, maximum is **1326.94**, and range is **1285.29** index points. Its mode is **100.00**, largely because all areas use the same 1980 Q1 base.

The mean exceeds the median, consistent with a distribution containing high index values. However, these pooled statistics mix places and dates. They are not average home prices, investment returns, or a national FHFA index. The standard deviation combines long-term price movement with geographic variation.

Every state code is tied as a mode because each appears 206 times. Quarter numbers are ordered within each year but repeat cyclically, so a quarter mean has little economic meaning. Many years are tied as modes because they each contain four quarters for every area. The final year contains only two quarters.

For a clearer recent comparison, the following table uses **2026Q2 year-over-year growth**:

| Statistic | YoY growth (%) |
| --- | --- |
| Mean | 3.44 |
| Median | 3.24 |
| Sample standard deviation | 1.75 |
| Minimum | -0.59 |
| Maximum | 8.70 |
| Range | 9.29 |

All latest growth values are unique, so there is no informative single mode. These statistics weight each area equally, including DC. They are not housing-stock-weighted national growth estimates.

**Highest growth areas:**

| state | hpi | yoy_pct |
| --- | --- | --- |
| HI | 982.57 | 8.70 |
| WV | 365.44 | 8.02 |
| IL | 565.87 | 5.99 |
| CT | 736.50 | 5.67 |
| WI | 646.66 | 5.60 |

**Lowest growth areas:**

| state | hpi | yoy_pct |
| --- | --- | --- |
| DC | 1,037.72 | -0.59 |
| CO | 853.19 | 0.20 |
| CA | 974.97 | 0.90 |
| AZ | 731.21 | 0.93 |
| WA | 1,033.33 | 1.05 |

In 2026Q2, **50 of 51 areas** had positive annual growth. Hawaii had the highest growth at **8.70%**, while Washington, DC had the lowest at **-0.59%**. Pennsylvania grew **4.65%**. These rankings describe this All-Transactions series and should not be confused with FHFA's flagship Purchase-Only rankings.

The calculation is `100 * (HPI_t / HPI_(t-4) - 1)`. Comparing the same quarter across years reduces seasonal comparison problems, but does not turn this series into seasonally adjusted data. Reindexing the complete quarterly calendar before calculating growth ensures that four periods mean four calendar quarters. The first four quarters for each area have no year-earlier comparison, producing **204 expected missing growth values**, separate from source-data missingness.

### Visualizations

The chart code below uses Pillow to keep dependencies small. Both figures are generated directly from the analyzed observations.


```python
# Create two reproducible PNG charts; coordinates map data values linearly.
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)
font = ImageFont.load_default(size=17)
small = ImageFont.load_default(size=14)

def canvas(title, subtitle):
    image = Image.new("RGB", (1100, 680), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 20), title, font=font, fill="#16324f")
    draw.text((45, 49), subtitle, font=small, fill="#555555")
    draw.text((45, 650), "Source: FHFA state All-Transactions HPI. Author calculations.", font=small, fill="#555555")
    return image, draw

# Rebase selected states to 2000 Q1 to compare growth over a shared interval.
selected = ["PA", "NY", "CA", "TX", "FL"]
trend = wide.loc[pd.Period("2000Q1", freq="Q"): , selected]
rebased = trend.div(trend.iloc[0]).mul(100)
image, draw = canvas("House price growth since 2000: five selected states", "Rebased to 2000 Q1 = 100; nominal index, not dollar prices")
x0, x1, y0, y1 = 90, 930, 100, 580
upper = int(np.ceil(rebased.max().max()/100)*100)
for value in range(0, upper+1, 100):
    y = y1 - value/upper*(y1-y0)
    draw.line((x0,y,x1,y), fill="#dddddd")
    draw.text((35,y-8), str(value), font=small, fill="#555555")
colors = ["#1864ab", "#e67700", "#2b8a3e", "#9c36b5", "#c92a2a"]
for state, color in zip(selected, colors):
    points=[(x0+i/(len(rebased)-1)*(x1-x0), y1-v/upper*(y1-y0)) for i,v in enumerate(rebased[state])]
    draw.line(points, fill=color, width=3)
    draw.text((960,120+selected.index(state)*30), state, font=font, fill=color)
for year in range(2000, int(rebased.index[-1].year)+1, 5):
    idx = rebased.index.get_loc(pd.Period(f"{year}Q1",freq="Q"))
    draw.text((x0+idx/(len(rebased)-1)*(x1-x0)-16,600),str(year),font=small,fill="#555555")
image.save(FIGURES/"state_trends.png")

# Full latest-quarter distribution: each point is one state/DC.
values = latest["yoy_pct"].dropna()
image, draw = canvas(f"Geographic spread of annual growth: {latest_period}", "Histogram; percentage change from the same quarter one year earlier")
counts, edges = np.histogram(values, bins=10)
left, right, top, bottom = 90, 1010, 110, 580
for tick in range(0,int(counts.max())+1,2):
    y=bottom-tick/(counts.max()+1)*(bottom-top)
    draw.line((left,y,right,y),fill="#dddddd")
    draw.text((55,y-8),str(tick),font=small,fill="#555555")
for i,count in enumerate(counts):
    xa=left+i/len(counts)*(right-left); xb=left+(i+1)/len(counts)*(right-left)
    draw.rectangle((xa+2,bottom-count/(counts.max()+1)*(bottom-top),xb-2,bottom),fill="#1864ab")
for i in range(0,len(edges),2):
    x=left+i/len(counts)*(right-left)
    draw.text((x-15,600),f"{edges[i]:.1f}",font=small,fill="#555555")
draw.text((460,625),"Year-over-year HPI growth (%)",font=small,fill="#555555")
draw.text((40,85),"Number of areas",font=small,fill="#555555")
image.save(FIGURES/"growth_histogram.png")
print("Saved figures/state_trends.png and figures/growth_histogram.png")
```

```text
Saved figures/state_trends.png and figures/growth_histogram.png
```

![House price growth in selected states](figures/state_trends.png)

These five states were chosen as illustrative comparisons, including Pennsylvania. Rebased indexes compare cumulative growth from a common date. The lines show that growth paths differ and that some markets experienced substantial declines after the mid-2000s. The selected states are not a representative national sample. An endpoint of 400 represents 300% cumulative growth since the rebasing date, not 400% growth.

![Distribution of annual growth](figures/growth_histogram.png)

The histogram summarizes all 51 areas in the latest quarter. Most areas show positive annual growth, but the spread indicates that geographic selection matters. A single cross-section does not establish which market will perform best in the future.

## Part 3: Expanding Investment Knowledge
### Additional Dataset: Zillow Observed Rent Index

A useful complementary dataset is the **Zillow Observed Rent Index (ZORI)**, available at [Zillow Research Housing Data](https://www.zillow.com/research/data/). ZORI measures typical observed market rent and is available for several geographic levels and housing categories.

FHFA helps describe historical price appreciation. Rent data add information about potential rental income and changes in rental demand. For example, rising home prices with slow rent growth could make rental income less attractive relative to acquisition cost. This is a question for further analysis, not a result established by the FHFA data alone.

To combine the datasets, use a shared geography, such as state-level series when available, standardize location names, and aggregate monthly rent observations to quarterly means. Restrict analysis to overlapping dates and compatible housing categories. If using metro-level rents instead, obtain FHFA metro indexes rather than treating state and metro boundaries as interchangeable. Normalize both series to a common start date to compare growth. A rent-to-HPI ratio is not a rental yield because HPI is an index rather than a dollar purchase price. Estimating a yield requires actual purchase values, rents, vacancy, and operating expenses.

This exploration provides a starting point for identifying markets to research further. It does not establish profitability. FHFA's mortgage-based coverage does not represent every housing transaction. State averages hide neighborhood variation. Refinance appraisals differ from completed sales, the data are revised, and the indexes are nominal rather than inflation-adjusted. Taxes, insurance, maintenance, financing, transaction costs, vacancy, and local employment also matter. Historical appreciation should not be treated as a forecast.

### Main Takeaway

The dataset is complete for the selected state-quarter panel. House price growth differs across locations. In 2026 Q2, 50 of the 51 areas had positive annual growth, with rates ranging from -0.59% in Washington, DC to 8.70% in Hawaii. These findings describe historical price movements. Rental income, purchase costs, and local conditions would also be needed to evaluate an investment.

## Part 4: Reproducing and Sharing the Analysis

1. Clone or download the full repository. Keep the notebook, `data/`, and `figures/` in the same project folder.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Open `Data_Understanding.ipynb` in Jupyter and run every cell in order from the project folder. The repository snapshot reproduces the saved results. In Colab, open the notebook and run all cells; with no snapshot present, it downloads the current CSV automatically. Set `USE_SNAPSHOT = False` to explicitly request a fresh download.
4. Confirm the printed time range, checksum, and data-quality results. New FHFA releases can change the results. If refreshing the report, regenerate the interpretation and README to match the new outputs.

## Sources

- FHFA, [House Price Index datasets](https://www.fhfa.gov/data/house-price-index), accessed October 25, 2026.
- FHFA, [HPI frequently asked questions](https://www.fhfa.gov/faqs/hpi), accessed October 25, 2026.
- Zillow Research, [Housing Data, ZORI](https://www.zillow.com/research/data/), accessed October 25, 2026. Suggested complementary data only; not analyzed here.
