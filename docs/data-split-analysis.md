# Data Split Analysis

This report compares the `baseline_1k` and `baseline_10k` training splits to determine whether the 10k set adds genuine diversity or mostly more repetition.

## baseline_1k

- rows: `800`
- unique companies: `712`
- unique fiscal periods: `45`
- unique event dates: `465`
- ticker entropy: `9.4116`
- date entropy: `8.6411`
- normalized-signature duplicate rate: `0.0000`
- average input length: `710.52` chars
- average output length: `109.33` chars

Direction distribution:

```json
{
  "negative": 273,
  "neutral": 274,
  "positive": 253
}
```

Magnitude distribution:

```json
{
  "flat": 63,
  "large": 163,
  "medium": 176,
  "small": 187,
  "unknown": 211
}
```

Top tickers:

```json
[
  {
    "key": "RTN",
    "count": 4,
    "share": 0.005
  },
  {
    "key": "BFI",
    "count": 3,
    "share": 0.0037
  },
  {
    "key": "BBBY",
    "count": 3,
    "share": 0.0037
  },
  {
    "key": "BHI1",
    "count": 3,
    "share": 0.0037
  },
  {
    "key": "SEGU",
    "count": 3,
    "share": 0.0037
  },
  {
    "key": "COMS",
    "count": 3,
    "share": 0.0037
  },
  {
    "key": "P",
    "count": 2,
    "share": 0.0025
  },
  {
    "key": "ARSW",
    "count": 2,
    "share": 0.0025
  },
  {
    "key": "PNCF",
    "count": 2,
    "share": 0.0025
  },
  {
    "key": "ACK",
    "count": 2,
    "share": 0.0025
  }
]
```

## baseline_10k

- rows: `10000`
- unique companies: `3888`
- unique fiscal periods: `59`
- unique event dates: `918`
- ticker entropy: `11.4644`
- date entropy: `9.3363`
- normalized-signature duplicate rate: `0.0000`
- average input length: `710.11` chars
- average output length: `109.3` chars

Direction distribution:

```json
{
  "negative": 3454,
  "neutral": 3507,
  "positive": 3039
}
```

Magnitude distribution:

```json
{
  "flat": 834,
  "large": 2065,
  "medium": 1981,
  "small": 2447,
  "unknown": 2673
}
```

Top tickers:

```json
[
  {
    "key": "INTC",
    "count": 23,
    "share": 0.0023
  },
  {
    "key": "MOT",
    "count": 20,
    "share": 0.002
  },
  {
    "key": "CSCO",
    "count": 20,
    "share": 0.002
  },
  {
    "key": "WFLT",
    "count": 17,
    "share": 0.0017
  },
  {
    "key": "RIG",
    "count": 16,
    "share": 0.0016
  },
  {
    "key": "BAC",
    "count": 16,
    "share": 0.0016
  },
  {
    "key": "NVLS",
    "count": 16,
    "share": 0.0016
  },
  {
    "key": "NOVL",
    "count": 15,
    "share": 0.0015
  },
  {
    "key": "APC",
    "count": 15,
    "share": 0.0015
  },
  {
    "key": "DH",
    "count": 15,
    "share": 0.0015
  }
]
```

## 1k vs 10k overlap

- overlapping event IDs: `14`
- new rows in 10k beyond 1k: `9986`
- new companies in 10k: `3298`
- new event dates in 10k: `468`
- new fiscal periods in 10k: `14`
- new normalized signatures in 10k: `9986`
- normalized-signature reuse rate in 10k: `0.0014`

## Split diagnosis

- The 10k split is genuinely broader than the 1k split, not just a larger duplicate pile.
- Exact input duplicates stayed at zero in both splits and normalized-signature reuse in 10k was only a tiny edge case.
- The 10k split added thousands of new companies and hundreds of new event dates, so the specialization effect is not explained by simple repetition alone.
- The more plausible explanation is prolonged exposure to one narrow structured task with the same output schema, which encourages task specialization even when row-level diversity increases.

