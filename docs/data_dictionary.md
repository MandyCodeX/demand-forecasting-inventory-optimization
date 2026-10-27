# 📖 Data Dictionary & Schema Specification

## Nationwide Demand Forecasting and Inventory Optimization

This document outlines the data schema, hierarchy structures, raw file layouts, and engineered feature definitions.

---

## 1. Raw M5 Files Schema

### 1.1 `calendar.csv`
Contains temporal metadata for 1,969 days from January 29, 2011 to June 19, 2016.

| Column | Type | Description | Example Values |
|---|---|---|---|
| `date` | `string` / `datetime` | Calendar date (YYYY-MM-DD) | `2015-01-01` |
| `wm_yr_wk` | `int64` | Walmart year-week ID | `11501` |
| `weekday` | `string` | Day name | `Saturday`, `Sunday` |
| `wday` | `int64` | Walmart day of week (1=Sat, 7=Fri) | `1` to `7` |
| `month` | `int64` | Month of year | `1` to `12` |
| `year` | `int64` | Year | `2011` to `2016` |
| `d` | `string` | Sequential day index | `d_1`, `d_2` ... `d_1941` |
| `event_name_1` | `string` (nullable) | Name of primary holiday / event | `SuperBowl`, `Thanksgiving` |
| `event_type_1` | `string` (nullable) | Category of event 1 | `Sporting`, `Cultural`, `National`, `Religious` |
| `event_name_2` | `string` (nullable) | Secondary event on same day | `Easter` |
| `event_type_2` | `string` (nullable) | Category of event 2 | `Religious` |
| `snap_CA` | `int8` | Binary flag if SNAP benefits active in California | `0` or `1` |
| `snap_TX` | `int8` | Binary flag if SNAP benefits active in Texas | `0` or `1` |
| `snap_WI` | `int8` | Binary flag if SNAP benefits active in Wisconsin | `0` or `1` |

---

### 1.2 `sell_prices.csv`
Contains the weekly selling price per store and product SKU.

| Column | Type | Description | Example Values |
|---|---|---|---|
| `store_id` | `string` | Store code | `CA_1`, `TX_2`, `WI_3` |
| `item_id` | `string` | SKU identifier | `FOODS_1_001` |
| `wm_yr_wk` | `int64` | Walmart year-week ID | `11501` |
| `sell_price` | `float32` | Retail selling price in USD ($) | `3.48` |

---

### 1.3 `sales_train_evaluation.csv` (or `sales_train_sample.csv`)
Wide format table containing historical unit sales per day.

| Column | Type | Description | Example Values |
|---|---|---|---|
| `id` | `string` | Unique series ID (`{item_id}_{store_id}_evaluation`) | `FOODS_1_001_CA_1_evaluation` |
| `item_id` | `string` | Product SKU | `FOODS_1_001` |
| `dept_id` | `string` | Department code | `FOODS_1`, `HOBBIES_2` |
| `cat_id` | `string` | High-level Category | `FOODS`, `HOUSEHOLD`, `HOBBIES` |
| `store_id` | `string` | Store code | `CA_1`, `TX_1`, `WI_1` |
| `state_id` | `string` | State code | `CA`, `TX`, `WI` |
| `d_1` ... `d_N` | `int32` | Daily unit sales on day $t$ | `0`, `3`, `12` |

---

## 2. Hierarchy Levels

The Walmart M5 dataset exhibits a strict 12-level hierarchical structure. Our project implements the 6 primary strategic tiers:

```
[Level 1] Nationwide Total (1 series)
    │
    ├── [Level 2] State (3 series: CA, TX, WI)
    │     │
    │     └── [Level 3] Store (10 series: CA_1..CA_4, TX_1..TX_3, WI_1..WI_3)
    │
    ├── [Level 4] Category (3 series: FOODS, HOUSEHOLD, HOBBIES)
    │     │
    │     └── [Level 5] Department (7 series: FOODS_1..3, HOUSEHOLD_1..2, HOBBIES_1..2)
    │
    └── [Level 6] Store x Item (Bottom Level: Individual product series)
```

---

## 3. Engineered Features Specification

| Feature Name | Category | Derivation Formula / Logic |
|---|---|---|
| `lag_7` | Autoregressive Lag | $y_{t-7}$ (Demand 7 days prior) |
| `lag_14` | Autoregressive Lag | $y_{t-14}$ (Demand 14 days prior) |
| `lag_21` | Autoregressive Lag | $y_{t-21}$ (Demand 21 days prior) |
| `lag_28` | Autoregressive Lag | $y_{t-28}$ (Demand 28 days prior) |
| `roll_mean_7` | Rolling Stat | $\frac{1}{7} \sum_{k=1}^7 y_{t-k}$ (Computed over shifted series) |
| `roll_mean_14` | Rolling Stat | $\frac{1}{14} \sum_{k=1}^{14} y_{t-k}$ |
| `roll_mean_28` | Rolling Stat | $\frac{1}{28} \sum_{k=1}^{28} y_{t-k}$ |
| `roll_std_7` | Rolling Stat | Sample standard deviation over recent 7-day lagged window |
| `roll_min_7` | Rolling Stat | Minimum demand observed in recent 7-day lagged window |
| `roll_max_7` | Rolling Stat | Maximum demand observed in recent 7-day lagged window |
| `sin_month` | Cyclical Calendar | $\sin(2\pi \cdot \text{month} / 12)$ |
| `cos_month` | Cyclical Calendar | $\cos(2\pi \cdot \text{month} / 12)$ |
| `sin_day_of_week`| Cyclical Calendar | $\sin(2\pi \cdot \text{day\_of\_week} / 7)$ |
| `cos_day_of_week`| Cyclical Calendar | $\cos(2\pi \cdot \text{day\_of\_week} / 7)$ |
| `is_weekend` | Calendar | Binary indicator ($1$ if Saturday or Sunday, else $0$) |
| `is_event` | Promotion | Binary flag ($1$ if national/cultural holiday, else $0$) |
| `snap_active` | Social Program | Active SNAP assistance flag matching row's `state_id` |
| `price_norm` | Price Dynamic | $\text{sell\_price} / \text{price\_max}$ |
| `price_discount`| Price Dynamic | $\max(0, 1 - \text{price\_norm})$ (Markdown percentage) |
| `is_discounted` | Price Dynamic | Binary flag ($1$ if discount ratio $> 5\%$, else $0$) |
