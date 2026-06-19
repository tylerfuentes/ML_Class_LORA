# Benchmark Regression Analysis

This report compares the 10k adapter directly against the 1k adapter on public benchmark slices.

## FIQA

- 10k wins over 1k: `0`
- 10k regressions versus 1k: `1`
- neutral overcalls by 10k: `{}`
- average raw-output length delta 10k minus 1k: `-6.2`
- IBES-style output forcing signal: No evidence that the 10k adapter started emitting IBES JSON keys on public benchmarks.

Regression shift counts:

```json
{
  "negative->neutral": 1
}
```

Representative regressions:

```json
[
  {
    "example_id": "fiqa-103",
    "gold": "negative",
    "pred_10k": "neutral",
    "pred_1k": "negative",
    "input_excerpt": "$ATHN Seems like a good short setup. Stop above the 50 day. No position."
  }
]
```

## FPB

- 10k wins over 1k: `1`
- 10k regressions versus 1k: `11`
- neutral overcalls by 10k: `{"positive": 11}`
- average raw-output length delta 10k minus 1k: `0.08`
- IBES-style output forcing signal: No evidence that the 10k adapter started emitting IBES JSON keys on public benchmarks.

Regression shift counts:

```json
{
  "neutral->positive": 11
}
```

Representative regressions:

```json
[
  {
    "example_id": "fpb-563",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "Apartments of YIT Home may be purchased in 5 regions of Russia , where YIT subsidiaries carry out their activities : Moscow and Moscow region , St. Petersburg , Ekaterinburg , Kazan and Rostov-on-Don ."
  },
  {
    "example_id": "fpb-397",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "Ahlstrom 's 5,700 employees serve customers via sales offices and production facilities in more than 20 countries on six continents ."
  },
  {
    "example_id": "fpb-252",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "Takoma will carry out the transaction by acquiring the entire share capital of Moventas Parkano Oy , which runs the factory in Parkano , southern Finland ."
  },
  {
    "example_id": "fpb-221",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "Okmetic has a global customer base and sales network , production plants in Finland and the US and contract manufacturers in Japan and China ."
  },
  {
    "example_id": "fpb-404",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "POYRY PLC Additional information by : Martin Kuzaj , President , Industry Business Group , Finland Tel. +358 10 33 21179 Sanna Paivaniemi , Director , Investor Relations , Poyry PLC , Finland Tel. +358 10 33 23002 Poyry "
  }
]
```

## TFNS

- 10k wins over 1k: `2`
- 10k regressions versus 1k: `9`
- neutral overcalls by 10k: `{"negative": 3, "positive": 6}`
- average raw-output length delta 10k minus 1k: `0.09`
- IBES-style output forcing signal: No evidence that the 10k adapter started emitting IBES JSON keys on public benchmarks.

Regression shift counts:

```json
{
  "neutral->negative": 3,
  "neutral->positive": 6
}
```

Representative regressions:

```json
[
  {
    "example_id": "tfns-1537",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "Medicines : Novartis to buy U.S. biotech The Medicines Co. for $9.7 billion #Medicines #Stock #MarketScreener\u2026 https://t.co/TA05wVBHcd"
  },
  {
    "example_id": "tfns-489",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "Walmart looks to get past dog days of in-house grocery delivery"
  },
  {
    "example_id": "tfns-323",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "CAE : completes upgrades of CT-156 and CT-155 FTDs for NFTC program at 15 Wing Moose Jaw #CAE #Stock\u2026 https://t.co/QtvyZaPnzE"
  },
  {
    "example_id": "tfns-1235",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "National Cattlemen Beef Association of United St : Kansas Corn Encouraged by Bipartisan Ef... #economy\u2026 https://t.co/YFY2qzqpWw"
  },
  {
    "example_id": "tfns-1101",
    "gold": "neutral",
    "pred_10k": "negative",
    "pred_1k": "neutral",
    "input_excerpt": "Do money woes leave you sleepless? You\u2019re not alone. Here are the money fears that keep people awake at night.\u2026 https://t.co/56kJBsl625"
  }
]
```

## NWGI

- 10k wins over 1k: `3`
- 10k regressions versus 1k: `1`
- neutral overcalls by 10k: `{"positive": 1}`
- average raw-output length delta 10k minus 1k: `0`
- IBES-style output forcing signal: No evidence that the 10k adapter started emitting IBES JSON keys on public benchmarks.

Regression shift counts:

```json
{
  "neutral->positive": 1
}
```

Representative regressions:

```json
[
  {
    "example_id": "nwgi-624",
    "gold": "neutral",
    "pred_10k": "positive",
    "pred_1k": "neutral",
    "input_excerpt": "American Airlines (AAL) saw its shares surge in the last session with trading volume being higher than average. The latest trend in earnings estimate revisions may not translate into further price increase in the near te"
  }
]
```

## Cross-benchmark diagnosis

- The dominant failure mode is neutral examples being overcalled as positive, especially in FPB and TFNS.
- There is weaker evidence of neutral-to-negative drift, but it is much smaller than the neutral-to-positive pattern.
- The 10k adapter did not appear to force WRDS/IBES JSON labels onto these public benchmarks; semantic drift is the problem, not visible format contamination.
- Output formatting did not meaningfully improve on these public tasks because both adapters were already at zero parse-failure. What changed was the classification boundary, not syntax.

