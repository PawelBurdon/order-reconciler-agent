# order-reconciler-agent

An AI agent that answers plain-English questions about the gaps between what
customers ordered and what was actually delivered, with every number computed
by a deterministic Python layer rather than by the model.

## Problem

Anyone who plans deliveries keeps two lists: what was promised and what
happened. They never agree. Some orders arrive short, some arrive late, some
arrive that nobody ordered, and some never arrive at all. Finding those gaps
means an outer join, a handful of derived columns and a pivot - twenty minutes
in a spreadsheet, repeated every week, and easy to get subtly wrong.

The questions people actually ask are not queries, though. They are sentences:
*which customers under-delivered in September?*, *what are the five biggest
shortfalls?*, *how bad is it overall?* This project puts a model in front of the
join so those sentences work, without letting the model anywhere near the
arithmetic.

## How it works

```
  "which customers under-delivered in September?"
                       |
                       v
        +--------------------------------+
        |  Gemini (function calling)     |  reads the tool schemas,
        +--------------------------------+  picks one and its arguments
                       |
                       |  function_call:
                       |  filter_records(status="QTY_MISMATCH",
                       |                 date_from="2025-09-01", ...)
                       v
        +--------------------------------+
        |  src/agent/tools.py            |  validates the arguments,
        +--------------------------------+  calls the core layer
                       |
                       v
        +--------------------------------+
        |  src/core/  (pandas)           |  outer join, diffs, aggregates
        +--------------------------------+
                       |
                       |  function_response: compact JSON -
                       |  totals over every match, plus at most
                       |  20 example rows
                       v
        +--------------------------------+
        |  Gemini                        |  calls another tool,
        +--------------------------------+  or writes the answer
                       |
                       v
   "In September 2025, three customers under-delivered a total of 385 units..."
```

The loop lives in `src/agent/agent.py`: send the question with the tool
declarations, and while the response contains function calls, execute them, put
the results back into the history and send again. A turn may contain several
calls at once; all of them are executed and all of the results travel back
together. The loop stops after eight rounds, so a model that keeps re-calling
the same tool fails visibly instead of quietly.

The SDK can run this loop for you - hand it Python callables instead of
declarations and it calls them itself. That is switched off on purpose here.

## Why two layers

The model decides *which* question to ask of the data. It never answers it.

Everything in `src/core/` is plain pandas with no AI in it: the join, the
differences, the percentages, the status of every line, the Excel export. The
tools in `src/agent/` are a thin wrapper that turns a function call into a call
on that layer and formats the result.

That split buys three things:

**The numbers are reproducible.** A language model asked to total a column will
produce a total. It will look right. Sometimes it is not, and there is no way to
tell from the output. Here the totals come out of pandas, so the same data
always gives the same answer and a unit test can pin it down.

**The context stays small.** A tool never returns the dataset - it returns
aggregates plus at most twenty example rows. On the sample data returning all 31
rows would work fine, but the habit does not survive 50,000 rows: the token bill
grows without limit and, worse, the model starts doing arithmetic by hand over a
wall of numbers and gets it wrong. The aggregates in every tool result are
computed over *all* matching rows, so a truncated list of examples never turns
into a wrong total. When rows do have to be dropped, they are sorted by the size
of the discrepancy first, so what falls off the end is the least interesting.

**Half the project works without the AI.** `report` runs the same core layer
with no model, no key and no network. `src/main.py` imports the agent package
inside the `ask` branch only, so that is a property of the code rather than a
claim in a readme.

The agent layer is isolated in `src/agent/`, so swapping the model provider
means changing a single module - the core layer would not notice.

## Example queries

These are real runs. A model does not answer identically twice, so the wording
of an answer will vary between invocations; the figures do not, because they
come out of pandas.

**1. A question that narrows the data**

```
$ python -m src.main ask "which customers under-delivered in September?"

In September 2025, three customers experienced under-deliveries (shortfalls):

- Crankset Supply Inc experienced an under-delivery of 60 units on order ORD-1023
  (SKU CHN-2002) delivered on September 3.
- Summit Bike Supply experienced an under-delivery of 45 units on order ORD-1025
  (SKU HDL-4004) delivered on September 9.
- Alpine Gear Co had an entire order missing (MISSING_ACTUAL) totaling 280 units
  on order ORD-1028 (SKU TYR-8008) planned for September 19.
```

`--verbose` shows how it got there:

```
[iteration 1]
  -> load_and_compare()
[iteration 2]
  -> filter_records(date_to='2025-09-30', date_from='2025-09-01')
[iteration 3]
  -> filter_records(date_from='2025-09-01', status='QTY_MISMATCH', date_to='2025-09-30')
[iteration 4]
  -> filter_records(date_from='2025-09-01', date_to='2025-09-30', status='MISSING_ACTUAL')
```

Three lookups: the whole month, then the two ways a line can come up short. The
280 units of the missing order are in there because a line that was never
delivered has no delivery date and would otherwise fall out of a September
filter - it is matched on the date it was promised for instead.

**2. A ranking question**

```
$ python -m src.main ask "top 5 largest shortfalls"

The top 5 largest shortfalls (by volume of missing units) across the 2025-07-03
to 2025-09-25 period are:

1. Alpine Gear Co (Order ORD-1028, SKU TYR-8008): 280 units short (100% missing,
   status MISSING_ACTUAL with a planned date of 2025-09-19).
2. Summit Bike Supply (Order ORD-1007, SKU DRL-7007): 75 units short (100%
   missing, status MISSING_ACTUAL with a planned date of 2025-07-21).
3. Crankset Supply Inc (Order ORD-1023, SKU CHN-2002): 60 units short (28.6%
   shortfall, status QTY_MISMATCH delivered on 2025-09-03).
4. Summit Bike Supply (Order ORD-1025, SKU HDL-4004): 45 units short (52.9%
   shortfall, status QTY_MISMATCH delivered on 2025-09-09).
5. Alpine Gear Co (Order ORD-1010, SKU CBL-1010): 40 units short (8.0%
   shortfall, status QTY_MISMATCH delivered on 2025-07-30).
```

This one exposes a gap in the tools rather than in the model. `top_discrepancies`
ranks by absolute size, so its top five contains a surplus of 300 units from an
unplanned delivery - which is a large deviation, but not a shortfall. The model
noticed, went back and rebuilt the ranking out of `filter_records` calls per
status:

```
[iteration 2]  -> top_discrepancies(limit=5, by='qty_diff')
[iteration 3]  -> filter_records(status='MISSING_ACTUAL')
[iteration 4]  -> filter_records(status='QTY_MISMATCH')
[iteration 5]  -> top_discrepancies(limit=5, by='qty_diff_pct')
```

The answer is right, at the cost of three round trips that a `direction`
argument on `top_discrepancies` would have saved. See the last section.

**3. An overview question**

```
$ python -m src.main ask "how many order lines have a discrepancy?"

Out of the 31 total order lines compared between July 3, 2025, and September 25,
2025, exactly 17 order lines have a discrepancy. This includes 9 lines with
quantity mismatches, 5 with date mismatches, 2 missing actual deliveries, and 1
unplanned delivery.
```

Two calls: `load_and_compare()`, then `get_summary()`. Every one of those
numbers is a field in the returned JSON; none of them were counted by the model.

## Stack

- Python 3.11+
- `google-genai` - the current Google SDK, using raw `FunctionDeclaration`
  schemas and a hand-written calling loop. No LangChain or any other agent
  framework.
- `gemini-3.5-flash-lite` by default; override with `--model`
- `pandas` for the comparison, `openpyxl` for the Excel export
- `pytest` for the tests, `python-dotenv` for the key
- No database. Two CSV files in, one Excel file or one answer out.

## Running locally

```bash
git clone <this repository>
cd order-reconciler-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The Excel report needs no key and no network:

```bash
python -m src.main report --output report.xlsx
```

```
Report written to /path/to/report.xlsx
31 order lines compared, 17 with a discrepancy (54.8%).
Net quantity difference: -253 units.
```

For the agent mode, get a key at https://aistudio.google.com/apikey and put it
in `.env`:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY
```

```bash
python -m src.main ask "which customers under-delivered in September?"
python -m src.main ask "top 5 largest shortfalls" --verbose
```

Both commands accept `--planned` and `--actual` to point at your own CSV files
instead of `sample_data/`.

Tests:

```bash
python -m pytest
```

They make no API calls: the tool tests build a comparison in memory and inject
it, so the suite runs without a key. The same suite runs in GitHub Actions on
Python 3.11 and 3.13, in a job with no key configured - which also builds the
Excel report, so the claim that half of this project works without the AI is
checked on every push rather than merely written down here.

### The sample data

Thirty planned lines and thirty actual lines of invented bicycle-parts orders
across three months, containing on purpose an order that was never delivered,
a delivery nobody planned, several quantity and date differences, one line
delivered in two shipments under the same identifier, and one blank delivery
date. The join key is `order_id` plus `sku`, because one order can contain
several products - and one of the sample orders does.

## What I'd do differently

**Let `top_discrepancies` take a direction.** It ranks by absolute size, so
shortfalls and surpluses compete in one list. Ask for the biggest shortfalls and
the model gets a surplus in the results, notices, and rebuilds the ranking out
of two `filter_records` calls - three extra round trips for something a
`direction="shortfall" | "surplus" | "any"` argument would have answered in
one. The model worked around a hole in the tool API, which is the most common
way an agent gets slow: the fix belongs in the schema, not in the prompt.

**Give the model a grouping tool.** There is no `group_by(dimension)`, so a
question like *which customers under-delivered* is answered by filtering per
customer or per status. It works on six customers; on sixty it would hit the
iteration limit. Same lesson as above - a tooling change, not a prompt change.

**Make the date comparison tolerant.** A delivery one day late and a delivery
thirty days late are both `DATE_MISMATCH`. A real reconciliation has a tolerance
window, and lateness should be rankable the way quantity is.

**Reconsider one status per line.** A line that is both short and late is
reported as `QTY_MISMATCH`; the slip survives in `date_diff_days`, but the
headline hides it. A list of statuses would be more honest, at the cost of a
column that is harder to filter on.

**Write an eval set for tool selection.** The tests cover what the tools do, not
whether the model picks the right one. What that needs is a fixture of
questions paired with the calls they should produce, run against the schemas -
which is also the thing that makes the tool descriptions safe to edit.

**Cache the comparison across runs.** It is recomputed on every invocation.
Fine for 31 rows, wasteful for a real extract; parquet next to the CSVs with a
staleness check would fix it.

**Support follow-up questions.** Every invocation is a fresh conversation. The
history is already threaded through the loop, so a REPL that keeps it between
questions is a small change - and would make the caching above matter more.
