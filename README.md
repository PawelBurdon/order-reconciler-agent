# order-reconciler-agent

An AI agent that answers plain-English questions about the gaps between what
customers ordered and what was actually delivered, with every number computed
by a deterministic Python layer rather than by the model.

## Quick look

- **What it is.** Two CSV files - orders as planned, orders as delivered - and
  a Gemini agent that answers questions about the difference between them by
  calling six Python functions. No LangChain: raw function calling and a loop
  written by hand.
- **Without an API key.** `pip install -r requirements.txt`, then
  `python -m src.main report --output report.xlsx` writes a three-sheet Excel
  comparison. No key, no network, no model - that half of the project does not
  depend on the AI at all.
- **With an API key.** Put a Gemini key in `.env`, then
  `python -m src.main ask "..."` for one question or `python -m src.main chat`
  for a conversation with follow-ups.

```
$ python -m src.main ask "how many order lines have a discrepancy?"

Out of the 31 total order lines compared between July 3, 2025, and September 25,
2025, exactly 17 order lines have a discrepancy. This includes 9 lines with
quantity mismatches, 5 with date mismatches, 2 missing actual deliveries, and 1
unplanned delivery.
```

Every one of those numbers came out of pandas and was handed to the model; none
of them were counted by it. Why that matters, how the loop works, and what the
six tools are: below.

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

## The tools

Six functions are declared to the model. It never sees anything else.

| Tool | What it returns |
| --- | --- |
| `load_and_compare()` | Reads both files and reconciles them. Reports the real customer names, the valid statuses and the period covered, so later calls use values that exist. |
| `get_summary()` | Aggregate figures for the whole comparison: lines matched, lines differing, the breakdown by status, planned against actual units, total under- and over-delivery. |
| `filter_records(customer, status, date_from, date_to)` | Totals computed over every matching line, plus at most 20 example rows. All arguments optional, combined with AND. |
| `top_discrepancies(by, limit)` | The largest deviations from the plan, ranked by absolute size. `by` is `qty_diff` or `qty_diff_pct`. |
| `group_by(dimension, sort_by, date_from, date_to, limit)` | Every customer, SKU or status as its own group with its own totals, worst first. |
| `generate_report(output_path)` | Writes the three-sheet Excel file and returns the path. |

The descriptions in the schema are the only thing the model reads when it
chooses, so each one says *when* to use the tool rather than only what it does,
and the pairs that are easy to confuse name each other: `filter_records` points
at `top_discrepancies` for "the biggest", `top_discrepancies` points back for
"filtered by customer or period", and `group_by` says outright never to loop
`filter_records` over customers.

## Why two layers

```
src/
  core/      loader.py  reconciler.py  report.py    plain pandas, no AI
  agent/     tools.py   agent.py       prompts.py   schemas, loop, prompt
  main.py                                           the CLI
```

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

- Alpine Gear Co under-delivered by 280 units (planned 380, actual 100 units
  across 2 order lines).
- Crankset Supply Inc under-delivered by 60 units (planned 335, actual 275
  units across 2 order lines).
- Summit Bike Supply under-delivered by 45 units (planned 85, actual 40 units
  across 1 order line).
```

`--verbose` shows how it got there:

```
[iteration 1]
  -> load_and_compare()
[iteration 2]
  -> group_by(sort_by='under_delivered_qty', date_to='2025-09-30',
              dimension='customer', date_from='2025-09-01')
```

One lookup. Every customer comes back in a single result, so there is no
per-customer loop to run. The 280 units of the never-delivered order are in
there because a line with no delivery date would otherwise fall out of a
September filter - it is matched on the date it was promised for instead.

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
unplanned delivery - a large deviation, but not a shortfall. The model notices,
and then has to assemble the ranking out of whatever else is available:

```
[iteration 2]  -> top_discrepancies(limit=5, by='qty_diff')
[iteration 3]  -> filter_records(status='MISSING_ACTUAL')
[iteration 4]  -> group_by(dimension='status', sort_by='under_delivered_qty')
[iteration 5]  -> filter_records(status='QTY_MISMATCH')
```

The five numbers come out the same on every run. The route does not: another
run reached them through `group_by(dimension='sku')` instead. That instability
is the symptom worth reading - when a question has no tool that answers it
directly, the model improvises, and improvisation is neither cheap nor
repeatable. A `direction` argument on `top_discrepancies` would collapse all of
this into one call. It is the first item on the roadmap for that reason.

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

**4. A conversation, where the second question leans on the first**

```
$ python -m src.main chat

> which customer is worst?

The customer worst served by under-delivered quantity between July 3, 2025, and
September 25, 2025, is Alpine Gear Co. They experienced a total under-delivered
quantity of 332 units (with a planned quantity of 1,100 units and an actual
delivered quantity of 768 units) spread across 3 discrepancy lines out of their
5 total order lines.

> and what exactly went wrong for them?

For Alpine Gear Co, five order lines were placed between July 3, 2025, and
September 25, 2025, two of which matched the plan perfectly.

The remaining three lines encountered issues:
- Order ORD-1028 (SKU TYR-8008, planned for September 19, 2025) resulted in a
  complete shortfall of 280 units as a MISSING_ACTUAL delivery.
- Order ORD-1010 (SKU CBL-1010, delivered July 30, 2025) had a quantity
  mismatch where 460 units were delivered instead of the planned 500.
- Order ORD-1016 (SKU SDL-5005, delivered August 19, 2025) was both late by 4
  days and under-delivered by 12 units.
```

*them* resolves because the previous exchange is still in the history. What is
not still in the history is the tool result behind the first answer - see the
next section.

## What the conversation remembers

`ask` is a single question. `chat` keeps a history, and what it keeps is a
deliberate choice, because an agent conversation grows in two very different
ways.

The questions and answers are short, and they are what a follow-up refers back
to - *them*, *that order*, *the same period*. The tool results are the opposite:
they are the bulkiest thing in the history by a wide margin, and by the next
question they are stale. So when an answer is finished, the exchange is
compacted down to the question and the answer, and the calls behind it are
dropped. If the model needs a number again it calls the tool again, which is
cheap and cannot go out of date.

The effect is that a long conversation costs roughly what a short one does,
instead of every turn being more expensive than the last. It also keeps the
history structurally simple - text in, text out - so there is no way to end up
sending a function call whose response was pruned, which the API rejects.

`/reset` clears it when the subject changes.

## Evaluating the tool selection

The test suite checks what the tools do. It cannot check the thing that
actually decides whether this works: whether the model reaches for the right
one. Those fail differently. A tool can be flawless and never be chosen, and
rewording a description to read better can quietly make the model choose worse
with nothing going red anywhere.

So there is an eval set - questions paired with what a good answer to them has
to satisfy - in `evals/cases.py`. Each case scores three things separately,
because "was the answer right" hides all three:

- **selection** - was the right tool reached for, and the wrong one left alone
- **efficiency** - how many calls it took to get there
- **grounding** - did the real figures end up in the answer, and did invented
  ones stay out

Cases assert what would be *wrong*, not what would be *different*: a required
tool, a call budget, figures that must appear. Never an exact sequence of
calls. The model is allowed to think differently on different days, and an eval
that goes red when it does is an eval that gets switched off within a week.

```bash
python -m evals.runner
python -m evals.runner --case worst_customer --verbose
```

```
Evaluating 9 cases against gemini-3.5-flash-lite

overview               pass   5/5 checks   2 calls
worst_customer         pass   4/4 checks   3 calls
customers_in_month     pass   5/5 checks   2 calls
biggest_shortfalls     GAP    6/7 checks   7 calls
    efficiency: 7 calls, at most 3 allowed
    known gap: top_discrepancies has no direction argument; roadmap item 1.
unplanned              pass   4/4 checks   2 calls
report                 pass   3/3 checks   2 calls
unknown_customer       pass   3/3 checks   2 calls
out_of_scope           pass   5/5 checks   1 calls
self_description       pass   3/3 checks   1 calls

9/9 cases passed, 1 known gap(s).
  selection   7/7
  efficiency  8/9
  grounding   23/23
```

Splitting the score by dimension is what makes that readable. Grounding is
23/23 - across nine questions, including one about a company that does not
exist and one about a column that does not exist, no figure was invented.
Selection is 7/7 - the right tool every time. The single miss is efficiency,
and it is the known gap: seven calls to assemble a ranking the schema cannot
express in one. A single overall percentage would have blurred a clean result
and a specific, already-diagnosed weakness into the same number.

Running the same set again is its own small result. Every stable figure
reproduces exactly - the same score, the same call counts on eight of the nine
cases. The ninth moves: `biggest_shortfalls` has cost seven calls and five. A
question the schema answers directly gets answered the same way twice; a
question it cannot gets improvised, and improvisation is not repeatable. That
variance is the gap making itself visible.

Two of the nine cases are worth pointing at. `unknown_customer` asks about a
company that is not in the data and `out_of_scope` asks for a figure the files
do not contain - the failure being watched for there is not a wrong answer but
a confident one. And `biggest_shortfalls` is marked as a known gap: it is run
and reported, but does not fail the build, because the reason it takes too many
calls is item 1 on the roadmap. Its call budget is already set to the number it
should reach once that lands, so closing the gap will be a measurement rather
than a claim.

The figures the cases expect are not typed in and trusted. `tests/test_evals.py`
recomputes each of them from the sample data with pandas and compares, so
editing the CSVs breaks the eval set loudly instead of making the model look
like it got worse.

That test runs on every push. The eval itself does not: it needs a key, it
costs money and it is not deterministic, so it is a separate CI job that runs
when a `GEMINI_API_KEY` secret is configured and says plainly that it skipped
when there is none.

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

Or keep the conversation open, so follow-up questions work:

```bash
python -m src.main chat
```

Somebody opening this for the first time does not know what it holds, so the
agent will explain itself. Ask *what is this and what can you tell me?* and it
describes the two files, the five statuses, the customers and period actually
present in your data, and what it cannot answer - there is no price, supplier
or stock information here, and it only reads. That is the single thing it is
allowed to answer without calling a tool, because it is a fact about the
program rather than a fact about the orders.

All three commands accept `--planned` and `--actual` to point at your own files
instead of `sample_data/`, and `ask` and `chat` also accept `--model`.

### Using your own data

Two CSV files, with exactly these columns:

```
planned_orders.csv    order_id, customer, sku, planned_qty, planned_date
actual_orders.csv     order_id, customer, sku, actual_qty,  actual_date
```

Dates are `YYYY-MM-DD`. Quantities are whole units. A missing date is
tolerated and surfaces as a data-quality flag rather than a discrepancy,
because a blank cell in a delivery date is a real thing that happens; a
missing `order_id`, `sku` or quantity is refused, because the comparison
cannot mean anything without them. Either way the loader names the file and
the line number instead of raising a traceback:

```
Data error: sample_data/actual_orders.csv: column 'actual_qty' contains
non-numeric value(s) in line(s) 14: 'twelve'.
```

The same `order_id` may appear on several lines when an order covers several
products - that is why the comparison keys on `order_id` plus `sku`. The same
`order_id` and `sku` twice is read as a split delivery: the quantities are
added up and the line is flagged `SPLIT_DELIVERY`.

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

## Roadmap

### Done since the first version

**A grouping tool.** `group_by` was the first entry in the list below. What
moved it up was a `--verbose` trace of *which customer has the worst
discrepancies?*: the model was calling `filter_records` once per customer,
seven iterations deep and still only three customers in, because nothing in
the schema could aggregate. It was compensating for a hole in the tool API.
With `group_by(dimension, sort_by, date range)` the same question takes two
calls, and *which customers under-delivered in September* went from four to
two. The numbers were identical before and after - only the route changed.

**A conversation mode.** `chat` keeps the history so follow-ups resolve, and
the interesting part is what it throws away. Each finished exchange is
compacted to the question and the answer; the tool traffic behind it is
dropped. That traffic is the bulk of the history and is stale by the next
question, while the talking is short and is what *them* or *that order* refers
back to. A long conversation therefore costs about what a short one does.

**An eval set for tool selection.** The tests could say the tools were correct
and nothing could say the model still chose them well. Nine cases now score
selection, efficiency and grounding separately, the figures they expect are
recomputed from the sample data rather than trusted, and the one case the tool
API cannot yet answer cleanly is marked as a known gap with its call budget
already set to the number it should reach when that gap closes. Details in the
section above.

**An agent that explains itself.** This one was never on the list. It came from
watching someone open the program and not know what to type. Asking *what is
this and what can you tell me?* now gets the real customers, the real period
and an explicit account of what the data does not contain. It is the one
answer allowed without a tool, and the prompt says why, so the exception
cannot quietly widen.

### Next, in the order I would do them

**1. Let `top_discrepancies` take a direction.** It ranks by absolute size, so
shortfalls and surpluses compete in one list. Ask for the biggest shortfalls
and the model gets a surplus in the results, notices, and reassembles the
ranking out of whatever else is available - by a different route on each run.
A `direction="shortfall" | "surplus" | "any"` argument answers it in one call.
This is first because it is the only item whose payoff is already measured: the
`biggest_shortfalls` eval case spends five to seven calls against a budget of
three, and both the budget and the variance are what the fix should remove.

**2. Filter the ranking tools the way `group_by` can be filtered.**
`top_discrepancies` covers the whole dataset while `group_by` takes a date
range, so *the biggest shortfalls in September* still has no single call that
answers it. Consistency between tools is part of a schema being usable.

**3. Make the date comparison tolerant.** A delivery one day late and one
thirty days late are both `DATE_MISMATCH`. A real reconciliation has a
tolerance window, and lateness should be rankable the way quantity is.

**4. Reconsider one status per line.** A line that is both short and late is
reported as `QTY_MISMATCH`; the slip survives in `date_diff_days`, but the
headline hides it. A list of statuses would be more honest, at the cost of a
column that is harder to filter on.

**5. Measure what the history compaction costs.** Dropping the tool results
between questions rests on an argument - a stale result is worth less than the
tokens it occupies - that is reasonable and untested. The way to know is the
eval harness above, pointed at follow-up questions answered with and
without the pruning.

**6. Cache the comparison across runs.** It is recomputed on every invocation.
Fine for 31 rows, wasteful for a real extract; parquet beside the CSVs with a
staleness check would fix it. Last because `chat` already reuses one comparison
for a whole session, so the waste is now per session rather than per question.

## License

MIT - see [LICENSE](LICENSE).
