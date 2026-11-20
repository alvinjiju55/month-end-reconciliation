# Month-end reconciliation

A payments company's books don't balance at month end. The platform has every
transaction it processed, the bank has what actually arrived, and the totals
differ. This tool finds out why, row by row, and builds a bridge from the
platform total to the bank total that ends in an **Unexplained** line of `0.000`.

It ships with a synthetic data generator (October 2026) that plants known
problems, so the reconciler can be tested against an answer key.

## What's in here

| File | What it does |
|---|---|
| `generator.py` | Builds reproducible fake data (fixed seed) with five switchable planted problems. Writes `platform_transactions.csv`, `bank_settlements.csv`, `ground_truth.json`. |
| `reconcile.py` | Pure pandas. Takes the two CSVs, flags every gap with a plain-English reason, builds the bridge. Never reads the ground truth. |
| `app.py` | Streamlit UI: generate or upload data, see totals, a waterfall of the bridge, flagged rows by type, download as CSV. |
| `tests/test_reconcile.py` | pytest suite that uses `ground_truth.json` as the answer key. |

## Run it

Python 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

**The app**

```bash
streamlit run app.py
```

Opens at http://localhost:8501. With no upload, it generates data in memory
from the seed and toggles in the sidebar, so it needs no files on disk.
To use your own data, pick "Upload my own CSVs" and give it both files.

**From the command line**

```bash
python generator.py                         # all plants on, writes the 3 files
python generator.py --disable duplicate     # switch one (or more) off
python generator.py --disable all           # clean data: books balance exactly
python generator.py --seed 7 --out-dir data # different seed, other folder

python reconcile.py                         # prints the bridge and sample flags
python reconcile.py --out-dir out           # also writes flags.csv, exceptions.csv, bridge.csv
```

**Tests**

```bash
python -m pytest -v
```

26 tests, about a minute. They generate their own data in temp folders and
never touch the files in the project folder.

## Input format

`platform_transactions.csv`: `trn_id, created_at, customer_id, type, amount, original_trn_id`
`bank_settlements.csv`: `settlement_id, batch_id, trn_ref, settled_on, amount`

## Gap types

| Type | What it means | Moves the totals? |
|---|---|---|
| **Duplicate** | Same `trn_id` recorded more than once in the platform; the bank only paid it once. Each extra copy overstates the platform. | Yes |
| **Never settled** | Platform transaction with no matching settlement anywhere in the bank extract. | Yes |
| **Settled after month end** | October transaction the bank settled in November. Timing only: the money did arrive. | Yes (for October) |
| **Truncation drift** | Bank keeps 2 decimals and drops the third. Under a cent per row, adds up over the month. | Yes, slightly |
| **Other amount mismatch** | Bank amount differs by more than truncation, or the bank settled one transaction more than once. Not planted by the generator, but detected. | Yes |
| **Bank only** | Bank settlement in the month with no platform transaction. Not planted, but detected. | Yes |
| **Orphan refund** (exception) | Refund whose `original_trn_id` doesn't exist. The money still left; nothing explains why. Listed separately because it doesn't change the totals. | No |

The bridge:

```
Platform October total
  less duplicates
  less never settled
  less settled after Oct 31
  less truncation drift
  +/- other mismatches
  + bank-only settlements
= Expected bank October total
  Actual bank October total
  Unexplained  (actual - expected, should be 0.000)
```

All money is summed as integer thousandths, so there's no floating-point
error in the totals.

## Assumptions

- All timestamps are UTC; month boundaries are midnight UTC.
- The period is October 2026. Platform side = transactions **created** in the month; bank side = settlements **dated** in the month.
- The bank truncates amounts to 2 decimals (toward zero); the platform keeps 3.
- The bank settles 1-2 days after the transaction, in daily batches.
- Matching is by `trn_id` = `trn_ref`, one to one. No fuzzy matching.
- Payments are positive, refunds negative, in both files.
- No fees, chargebacks, FX, or reversals: the bank pays the gross amount.

## What this would get wrong in production

- **Real banks net and batch.** Settlements usually arrive as one net payout per batch with fees taken off, not one line per transaction with a clean `trn_ref`. You'd need to reconcile at batch level first, then drill down.
- **Fees, chargebacks, reserves, FX** are all ignored. Each would land in "Unexplained" or be misread as a mismatch.
- **Exact-ID matching only.** A missing or mistyped reference shows up as one "never settled" plus one "bank only" instead of a single near-match. Production needs fuzzy matching on amount, date and customer, with a review queue.
- **Truncation is assumed.** If the bank rounds instead, or rounds at batch level, every row would be flagged as a mismatch.
- **"Never settled" depends on the extract window.** Anything still in flight after the bank file ends looks unsettled. It needs ageing (how many days overdue), not a yes/no.
- **Duplicates mean identical `trn_id` only.** Two different IDs for the same real payment (a retry, a double click) wouldn't be caught.
- **Orphan refund checks only this file.** A refund of a September payment would be flagged as an orphan; production would look up history.
- **Previous month's tail is ignored.** September transactions settling on Oct 1-2 would show as "bank only" because the platform file only has October.
- **Everything is in memory**, fine for thousands of rows, not for millions. No audit trail, sign-off, or persistence of decisions.
- **Time zones:** a bank that dates settlements in local time would shift some rows across the month boundary.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (public, or private with access granted).
2. At https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**.
3. Pick the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings**, choose Python **3.10**.
5. Deploy. The app installs `requirements.txt` and generates its data in memory, so no data files need to be committed.
