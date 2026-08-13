# BTPR3203-Python-for-Data-Science-2026B-Case-Study

## Sustainable Industrial Activity in Malaysia: Water Consumption and Industrial Production

### 1. Project Overview

This case study analyses sustainable industrial activity in Malaysia using two public datasets from the Department of Statistics Malaysia (DOSM) through Malaysia's open data platform, **data.gov.my**.

The analysis focuses on:

* Geographical differences in non-domestic water consumption
* Long-term changes in non-domestic water consumption
* The relationship between industrial production and non-domestic water demand
* Implications for sustainable industrial and water-resource management

---

## 2. Research Questions

### RQ1 – Data Preparation

**How can the raw Malaysian state-level water consumption data from 2003 to 2022 and national Industrial Production Index (IPI) data from 2015 to 2022 be cleaned, restructured, and transformed to produce reliable indicators for sustainable industrial analysis?**

### RQ2 – Data Analysis

**Which Malaysian states had the highest non-domestic water consumption in 2022, how did their consumption change from 2003 to 2022, and what was the strength of the association between national non-domestic water consumption and industrial production from 2015 to 2022?**

### RQ3 – Data Visualisation

**What geographical and temporal patterns in Malaysian non-domestic water consumption are revealed by visualisations from 2003 to 2022, and what overall pattern is visible between national non-domestic water consumption and industrial production from 2015 to 2022?**

---

## 3. Datasets

### 3.1 Water Consumption Dataset

**File:** `water_consumption.csv`

**Source:** Department of Statistics Malaysia (DOSM), data.gov.my

**Period:** 2003–2022

**Records:** 600

**Variables:**

| Variable | Description                                           |
| -------- | ----------------------------------------------------- |
| `state`  | Malaysian state or national total labelled `Malaysia` |
| `sector` | `domestic` or `nondomestic`                           |
| `date`   | Annual reporting date                                 |
| `value`  | Water consumption in million litres per day (MLD)     |

The dataset is used to analyse geographical and temporal patterns in non-domestic water consumption. National `Malaysia` observations are separated from state-level observations when conducting geographical comparisons.

### 3.2 Industrial Production Index Dataset

**File:** `ipi.csv`

**Source:** Department of Statistics Malaysia (DOSM), data.gov.my

**Period used:** 2015–2022

**Records:** 398

**Variables:**

| Variable   | Description                             |
| ---------- | --------------------------------------- |
| `series`   | IPI series type                         |
| `date`     | Monthly reporting date                  |
| `index`    | Industrial Production Index, 2015 = 100 |
| `index_sa` | Seasonally adjusted IPI, 2015 = 100     |

Only the `abs` series is used. Monthly observations are aggregated into annual mean IPI values before being merged with national non-domestic water consumption.

---

## 4. Methodology

The analysis follows the workflow below:

**Raw datasets → Data preparation → Transformation → Analysis → Visualisation → Findings → Recommendations**

### Step 1 – Data Preparation

The datasets are examined for:

* Structure and data types
* Descriptive statistics
* Missing values
* Duplicate records
* Data consistency
* Invalid dates

The following transformations are then performed:

* Convert dates to datetime format
* Extract year
* Separate national and state-level water data
* Pivot domestic and non-domestic sectors
* Calculate total water consumption
* Calculate non-domestic water share
* Select the IPI `abs` series
* Aggregate monthly IPI to annual means
* Merge water and IPI data for 2015–2022

### Step 2 – Analytical Operations

Three analytical operations are conducted:

1. **2022 State-Level Comparison**
   States are ranked according to non-domestic water consumption and their percentage contribution is calculated.

2. **2003–2022 Growth Analysis**
   Percentage growth in non-domestic water consumption is calculated for each state.

3. **Pearson Correlation Analysis**
   The linear association between annual national non-domestic water consumption and annual mean IPI is measured for 2015–2022.

### Step 3 – Visualisation

Three visualisations are generated:

1. **Figure 1:** Non-Domestic Water Consumption by State (2022)
2. **Figure 2:** Non-Domestic Water Consumption Growth by State (2003–2022)
3. **Figure 3:** National Non-Domestic Water Consumption vs Industrial Production Index (2015–2022)

Each visualisation is selected to communicate a specific geographical, temporal, or resource-production finding more clearly than a table alone.

---

## 5. Key Findings

### Geographical Concentration

Selangor recorded the highest non-domestic water consumption in 2022 at **1,350 MLD**, followed by Johor (**536 MLD**) and Sarawak (**463 MLD**). The five highest-consuming states accounted for approximately **66.9%** of state-level non-domestic water consumption.

### Long-Term Growth

Between 2003 and 2022, non-domestic water consumption increased by **117.9% in Johor** and **100.6% in Selangor**, indicating substantial long-term growth in non-domestic water demand in several states.

### Industrial Production and Water Consumption

The Pearson correlation coefficient was **r = 0.484**, indicating a moderate positive linear association between national non-domestic water consumption and industrial production during 2015–2022.

A notable decline occurred in 2020, coinciding with the COVID-19 pandemic and Malaysia's Movement Control Order (MCO), which disrupted economic and industrial activities.

The correlation represents an association and does not establish causation.

---

## 6. Recommendations

The findings support three main recommendations:

### Recommendation 1

**Stakeholders:** State water authorities and state economic development agencies

Prioritise water-efficiency measures in high-consumption states, particularly Selangor, Johor, Sarawak, Pulau Pinang, and Perak.

### Recommendation 2

**Stakeholders:** Ministry of Investment, Trade and Industry (MITI) and state economic development agencies

Incorporate water-efficiency considerations into industrial development planning, particularly in states with substantial long-term growth in non-domestic water consumption.

### Recommendation 3

**Stakeholders:** Department of Statistics Malaysia (DOSM) and relevant industrial policymakers

Develop a regular monitoring framework combining industrial production and water-consumption indicators to support resource-efficiency planning.

---

## 7. Project Structure

```text
BTPR3203_CaseStudy/
│
├── README.md
├── case_study.py
├── case_study.ipynb
│
├── water_consumption.csv
├── ipi.csv
│
├── cleaned_water_consumption_by_state.csv
├── cleaned_national_water_vs_ipi.csv
│
├── fig1_nondomestic_water_by_state_2022.png
├── fig2_nondomestic_water_growth_2003_2022.png
└── fig3_national_water_vs_ipi.png
```

---


## 8. Output Files

The pipeline generates the following files:

### Cleaned Datasets

* `cleaned_water_consumption_by_state.csv`
* `cleaned_national_water_vs_ipi.csv`

### Visualisations

* `fig1_nondomestic_water_by_state_2022.png`
* `fig2_nondomestic_water_growth_2003_2022.png`
* `fig3_national_water_vs_ipi.png`

---

## 9. Python Libraries

The project uses the following Python libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

* **Pandas** – data loading, cleaning, transformation, aggregation, and analysis
* **NumPy** – numerical operations and feature engineering
* **Matplotlib** – visualisation and figure generation
* **Seaborn** – statistical and categorical visualisation

---

## 10. References

Department of Statistics Malaysia. (2020, June 11). *Index of industrial production, Malaysia, April 2020*. https://www.dosm.gov.my/portal-main/release-content/index-of-industrial-production-malaysia-april-2020

Department of Statistics Malaysia. (2026). *Industrial Production Index (IPI)* [Data set]. data.gov.my. https://data.gov.my/data-catalogue/ipi

Department of Statistics Malaysia. (2024). *Water consumption by state and sector* [Data set]. data.gov.my. https://data.gov.my/data-catalogue/water_consumption
