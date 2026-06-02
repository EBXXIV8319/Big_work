# Equipment Loss Forecasting Context

This context defines the domain language for decomposing and forecasting Russia-Ukraine war equipment-loss time series in this assignment.

## Language

**Equipment loss**:
An observed or claimed loss count for a military equipment category on a specific date.
_Avoid_: casualty, death, prisoner exchange

**Cumulative loss series**:
A time series where each record contains total losses accumulated up to that date.
_Avoid_: daily losses

**Daily loss series**:
A time series obtained by differencing a cumulative loss series across consecutive dates.
_Avoid_: raw cumulative count

**Trend component**:
The slowly varying baseline level in the daily loss series.
_Avoid_: seasonal pattern, anomaly

**Seasonal component**:
A repeating calendar pattern in the daily loss series, especially a weekly reporting rhythm.
_Avoid_: long-term trend

**Shock component**:
A residual spike after trend and seasonal effects have been removed.
_Avoid_: trend change

**Forecast interval**:
An uncertainty band around future predicted daily losses.
_Avoid_: deterministic prediction

## Relationships

- A **Cumulative loss series** is transformed into a **Daily loss series** by differencing.
- A **Daily loss series** is decomposed into **Trend component**, **Seasonal component**, and **Shock component**.
- A **Forecast interval** belongs to a model forecast for a **Daily loss series**.

## Example dialogue

> **Analyst:** "Do we model the JSON equipment counts directly?"
> **Domain expert:** "No. Those counts are cumulative, so we first convert them into a daily loss series."

## Flagged ambiguities

- The local Excel file describes casualties and prisoner exchanges, not **Equipment loss**. The assignment therefore uses the downloaded equipment-loss JSON as the primary data source and keeps the Excel file only as a data-audit note.
