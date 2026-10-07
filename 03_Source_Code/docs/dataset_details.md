# Dataset Details — 5G Network Operations (NetOps) Dataset

| Item | Value |
|---|---|
| Name | 5G Network Operations (NetOps) Dataset |
| Source | Hugging Face — https://huggingface.co/datasets/fenar/5g_netops (file `5G_netops_data_100K.csv.xz`) |
| Alternative | RANalyzer Dataset — https://github.com/wineslab/RANalyzer-Dataset (not used) |
| Format | CSV (xz-compressed), loaded with `pandas.read_csv("hf://datasets/fenar/5g_netops/5G_netops_data_100K.csv.xz")` |
| Size | 100,000 records × 27 columns; no duplicates; no missing values |
| Nature | Synthetic network-operations records (about 38K distinct city names, 59 state/territory codes) |
| Local copy | `03_Source_Code/data/raw/5G_netops_data_100K.csv` (downloaded automatically by `python -m data.ingest`) |
| Sample | `06_Dataset_Sample.csv` (first 200 rows plus derived fields) |

## Key fields

<!-- DATASET_FIELDS -->

## Derived fields (created by the ingestion pipeline)

| Field | Description |
|---|---|
| `case_id` | Stable identifier `NOC-000000` … `NOC-099999` |
| `c_<condition>` (21 columns) | 0 = normal, 1 = degraded (worse than fleet P75/P25), 2 = severe (P90/P10) |
| `conditions` | Comma-separated list of the conditions that are present |
| `category`, `category_name` | Incident taxonomy class (8 categories + Normal) |
| `risk_score` | 0–100 composite (fault likelihood 50 %, degradation breadth 30 %, customer impact 20 %) |
| `priority`, `severity` | P1 Critical / P2 Major / P3 Minor / P4 Low |
| `text` | Incident narrative used for semantic retrieval and LLM grounding |

## Condition thresholds (computed from the data)

<!-- DATASET_THRESHOLDS -->

## Distribution of derived labels

<!-- DATASET_DISTRIBUTION -->

## Data-quality findings

- `Issue Reported` is about 50/50 and independent of all KPIs (identical group means), so it is treated as noise and
  used only as a filter.
- `Fault Occurrence Rate` is strongly determined by Critical Alarm Count (r = 0.93), Authentication Failures (0.64)
  and Call Drop Rate (0.46). Throughput is strongly anti-correlated with Critical Alarm Count (r = −0.96).
- Most other KPIs are mutually independent (uniform-like synthetic generation), which limits the achievable accuracy
  of the performance models (see `model_metrics.json`).
- Season has a mild effect: Winter has a higher fault rate (41.9 % vs about 37 %), and Summer has higher latency.

## Example narrative (record NOC-000000)

<!-- DATASET_EXAMPLE -->
