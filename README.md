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

Before an answer is handed back, it is checked against what the model was
actually shown: every identifier-shaped token in it - an order id, a SKU - has
to appear in the question, in the conversation so far, or in a tool result from
this turn. One that does not was invented, and the model is told exactly which
one and asked again. This exists because it happened; the evaluation section
has the case.

The SDK can run this loop for you - hand it Python callables instead of
declarations and it calls them itself. That is switched off on purpose here.

## The tools

Six functions are declared to the model. It never sees anything else.

| Tool | What it returns |
| --- | --- |
| `load_and_compare()` | Reads both files and reconciles them. Reports the real customer names, the valid statuses and the period covered, so later calls use values that exist. |
| `get_summary()` | Aggregate figures for the whole comparison: lines matched, lines differing, the breakdown by status, planned against actual units, total under- and over-delivery. |
| `filter_records(customer, status, date_from, date_to)` | Totals computed over every matching line, plus at most 20 example rows. All arguments optional, combined with AND. |
| `top_discrepancies(by, limit, direction, date_from, date_to)` | The largest deviations from the plan. `by` is `qty_diff`, `qty_diff_pct` or `date_diff_days`; `direction` narrows to one side - `shortfall`/`surplus` for quantities, `late`/`early` for dates; the dates narrow the period. |
| `group_by(dimension, sort_by, date_from, date_to, limit)` | Every customer, SKU or status as its own group with its own totals, worst first. |
| `generate_report(output_path)` | Writes the three-sheet Excel file and returns the path. |

The descriptions in the schema are the only thing the model reads when it
chooses, so each one says *when* to use the tool rather than only what it does,
and the pairs that are easy to confuse name each other: `filter_records` points
at `top_discrepancies` for "the biggest", `top_discrepancies` points back for
"filtered by customer", and `group_by` says outright never to loop
`filter_records` over customers. Where two tools accept the same argument they
describe it the same way, because a date range that means one thing in one
schema and another elsewhere is a trap the model has no way to see.

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

The 5 largest shortfalls by volume across the period from July 3, 2025, to
September 25, 2025, are:

1. Alpine Gear Co (Order ORD-1028, SKU TYR-8008): A shortfall of 280 units
   (100% missing, planned for September 19, 2025).
2. Summit Bike Supply (Order ORD-1007, SKU DRL-7007): A shortfall of 75 units
   (100% missing, planned for July 21, 2025).
3. Crankset Supply Inc (Order ORD-1023, SKU CHN-2002): A shortfall of 60 units
   (actual delivery of 150 units against 210 planned on September 3, 2025).
4. Summit Bike Supply (Order ORD-1025, SKU HDL-4004): A shortfall of 45 units
   (actual delivery of 40 units against 85 planned on September 9, 2025).
5. Alpine Gear Co (Order ORD-1010, SKU CBL-1010): A shortfall of 40 units
   (actual delivery of 460 units against 500 planned on July 30, 2025).
```

```
[iteration 1]  -> load_and_compare()
[iteration 2]  -> top_discrepancies(direction='shortfall', limit=5, by='qty_diff')
```

Two calls - but this example earns its place because until recently it took
five, or seven, depending on the run.

`top_discrepancies` used to rank by absolute size and nothing else. Its top
five therefore contained a 300-unit surplus from an unplanned delivery: a large
deviation, but not a shortfall. The model noticed every time, and went off to
rebuild the ranking from whatever else was to hand - `filter_records` per
status on one run, `group_by(dimension='sku')` on the next. The five figures
came out right every time. The number of calls did not, and that instability
was the real signal: a question the schema cannot express gets improvised, and
improvisation does not repeat.

The fix was a `direction` argument, not a better prompt. The model had been
choosing correctly all along - it was choosing among tools that could not say
what it needed. It now reaches for `direction='shortfall'` unprompted, and the
eval case guarding this has come in at two calls on three consecutive runs. The
before and after are in the evaluation section.

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
Evaluating 12 cases against gemini-3.5-flash-lite

overview               pass   5/5 checks   2 calls
worst_customer         pass   4/4 checks   2 calls
customers_in_month     pass   5/5 checks   2 calls
biggest_shortfalls     pass   7/7 checks   2 calls
shortfalls_in_month    pass   7/7 checks   2 calls
latest_deliveries      pass   5/5 checks   2 calls
every_late_or_early_line pass 7/7 checks   2 calls
unplanned              pass   4/4 checks   2 calls
report                 pass   3/3 checks   2 calls
unknown_customer       pass   4/4 checks   2 calls
out_of_scope           pass   5/5 checks   1 calls
self_description       pass   3/3 checks   1 calls

12/12 cases passed.
  selection   10/10
  efficiency  12/12
  grounding   37/37
```

That run predates the two conversation cases added for the compaction
measurement; the set is fourteen now, and the scorecard will be refreshed the
next time there is quota to run all of it at once.

Splitting the score by dimension is what makes that readable. Grounding is
37/37 - across twelve questions, including one about a company that does not
exist and one about a column that does not exist, no figure was invented.
Selection is 10/10 and efficiency is 12/12: every question answered inside its
call budget. A single overall percentage would say none of that, and would have
hidden the runs where it was not true.

### What it has actually caught

Two things, neither of which a person reading the answers would have noticed.

**A ranking that improvised.** Before `top_discrepancies` took a direction,
`biggest_shortfalls` cost five calls on one run and seven on the next, arriving
at the same five correct figures by a different route each time. The answer was
never wrong, so nothing looked broken. The variance was the signal, and it is
what item 1 of the roadmap was, and why it went first.

**An order id quietly corrupted.** On one run the `unplanned` case answered:
*order `ORD-1000` (SKU CBL-1010) for Northwind Cycles arrived with 300
unplanned units on 2025-09-25*. Everything in that sentence is right except the
identifier - the order is `ORD-1090`, and the correct value was sitting in the
tool result the model was reading from. An answer that is right about the
quantity, the product, the customer and the date passes a human review; this
one would have sent somebody looking for an order that does not exist. It
appeared roughly once in five runs, which is exactly the failure rate that no
amount of trying things by hand will find.

The first attempt at a fix was a line in the system prompt telling the model to
copy identifiers character for character. Four runs passed, which proved
nothing at that failure rate and was described that way at the time. Two
commits later it happened again, in CI, on the same order.

A prompt is a request. So the answer is now checked before it is returned:
every identifier-shaped token in it is looked for in the question, in what the
model has already said, and in every tool result it received. Anything else was
invented, and the model is told so and asked once more, quoting the token back
to it. If the second answer still contains it, the answer carries a visible
`[Unverified: ...]` note rather than reading cleanly. This is the same
principle as the rest of the project, applied one step further: where Python
can check the model, it should.

Eight runs since, all clean, and the correction has not yet fired live - which
again proves little, because eight clean runs at one-in-six are a 23% coincidence.
What is certain is different in kind: the failure is now caught by construction
whenever it happens, which the unit tests demonstrate against a stubbed model
without spending a token.

That second finding also improved the harness: a failing case now prints the
answer and the calls that produced it. A red eval in CI that cannot be
diagnosed without re-running it by hand is one nobody diagnoses, and the first
question to ask of any failure is whether the model was wrong or the assertion
was. It has been worth asking: of the five red runs so far, three were the
assertion rather than the model.

`unknown_customer` looked for a phrase meaning "no". It failed a correct answer
worded differently, then passed on a sentence about pricing that happened to
contain one. A list of phrases cannot express "it denied the company exists" -
there are endless ways to say no. It became a regex for a negation in the same
sentence as the name, with a unit test over three real answers and two
fabrications, including the near-miss that had slipped through.

`latest_deliveries` then failed a flawless answer because it wanted the order
id and the delay in one sentence and the model used two. That is the general
form of the same mistake: asserting how close two things sit in prose is
asserting the model's punctuation. It now looks for "4 days", which identifies
the right line because exactly one line in the data moved by four days - and a
test guards that fact, so the day a second one does, the assertion fails loudly
instead of quietly meaning less.

Two cases are conversations rather than single questions: a follow-up is asked
on the same thread, and only the last answer is scored while the calls are
counted across both turns. They exist because they are the only place the
history compaction has any effect at all, and running the set with
`--keep-history` puts the same questions the other way.

Two more are worth pointing at: `unknown_customer` asks about a
company that is not in the data, `out_of_scope` asks for a figure the files do
not contain. The failure being watched for in both is not a wrong answer but a
confident one.

A case can also be marked as a known gap, in which case it runs and is reported
but does not fail the build. `biggest_shortfalls` carried that mark while the
tool API could not express its question, with its call budget already set to
the number the fix was supposed to reach - so closing the gap produced a
measurement rather than a claim. The mark is gone now and the budget is live.
A red build for something already written down in the roadmap teaches nobody
anything; a red build for a regression is the point.

The figures the cases expect are not typed in and trusted. `tests/test_evals.py`
recomputes each of them from the sample data with pandas and compares, so
editing the CSVs breaks the eval set loudly instead of making the model look
like it got worse.

That test runs on every push. The eval itself does not: it needs a key, it
costs money and it is not deterministic, so it is a separate CI job that runs
when a `GEMINI_API_KEY` secret is configured and says plainly that it skipped
when there is none.

It also has a quota, and the run that first hit it is worth recording. A case
that cannot be reached is reported as `ERROR`, not `FAIL`, because "the model
answered badly" and "nobody could ask the model" are different results and
only the first says anything about the code. Both still fail the build:
missing signal is not the same as a pass, and an eval that goes green when it
did not run is worse than no eval. A dozen cases at two or three calls each is
roughly twenty-five requests per push, against a free-tier ceiling of five
hundred a day, so it is a limit a heavy day of development reaches and normal
use does not. The workflow can be re-run from the Actions tab once the quota
resets.

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

**An empty result that says why it is empty.** Found by running the eval
against a different model when the usual one had run out of quota. Asked about
September, it guessed the year as 2023, got nothing back, and reported that no
customer had under-delivered - true of September 2023, and not the question. An
empty result is ambiguous between "nothing went wrong then" and "there is no
then", and only one of those is worth telling somebody. Every tool that takes a
date range now says which range exists when a filter matches nothing. The model
was not at fault; the tool gave it no way to tell the two apart.

**Half an answer on what the history compaction costs.** `chat` throws tool
results away between questions on the argument that a stale result is worth
less than the tokens it occupies. That was reasonable and untested, so the
runner gained `--keep-history` and the case set gained two conversations, and
the same questions were put both ways.

The cost side is settled: keeping everything made the context 58% larger on
average and 90% larger at its peak. The accuracy side is not. One round, two
cases: compacted answered both, full history missed a figure on one. That is a
single run and one failure is well inside the noise, so it is written here as
what it is rather than as a result. Repeating it properly is the one item left
on the list below, and the harness for it now exists. The measurement also had
to run on `gemini-2.5-flash` rather than the default, because a day of
developing against the eval had spent the daily quota - itself a fact worth
knowing about running evals on a free tier.

**More than one status per line.** The follow-up to the item below, and the
one where the fix went into the data rather than the schema. Asked to list
every order that arrived on a different date, the model filtered on the
`DATE_MISMATCH` status and returned five of the six, on all three runs. The
missing one was short as well, and quantity won the single status field. There
was already a warning about this in a tool description - on the wrong tool, so
it changed nothing, which is the second time in this project that a prompt has
turned out to be a request rather than a guarantee.

A line now carries every status that applies to it, and filtering matches any
of them; the single `status` field remains as the headline a report column has
room for. Three runs before, five of six; three after, six of six - **by the
same route**. The model did not learn anything. The obvious path stopped being
a trap.

**A way to rank by lateness.** The measurement that found a wrong answer
rather than a slow one. Asked which deliveries were most late, the model
filtered on the `DATE_MISMATCH` status - the only route there was - and named
the wrong orders on all three runs, fluently, with the right SKUs and dates
attached. The latest delivery in the data is not a `DATE_MISMATCH`: it was
short as well, and quantity wins the status. So the obvious route silently
skips the worst case, and the answer reads as complete.

`top_discrepancies` now ranks by `date_diff_days` with `direction` of `late`
or `early`, off the column rather than the status. Three runs before, all
wrong; three after, all right, the model reaching for it unprompted. The gap
this exposed in the status design is item 2 below - ranking makes the question
answerable, it does not stop the status from misleading elsewhere.

**A date range on the ranking tool.** The item where measuring first changed
the answer. The prediction was that "the biggest shortfalls in September" would
be expensive, because ranking and filtering lived in different tools. It was
not: three calls, the same route every run, the right figures every time. What
the trace showed instead was the model pulling a month from one tool and a
ranking from another and intersecting them in its head - getting it right, but
by being careful rather than by construction. That is the argument for the fix,
and it is a different argument from the one predicted: not speed, but a wrong
answer that becomes impossible instead of merely unlikely. Three calls to two,
and one fewer thing depending on the model paying attention.

**A direction on the ranking tool.** The first item to be promoted off the list
below by a number rather than a hunch. `top_discrepancies` ranked by absolute
size, so "the biggest shortfalls" returned a surplus at the top and the model
spent five to seven calls - a different route each run - assembling the answer
from other tools. A `direction` argument closed it: the eval case came in at
two calls, three runs in a row, with the model finding the argument on its own.
The measurement existed before the fix did, which is the only reason the
improvement is a fact rather than an impression.

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

**1. Make the date comparison tolerant.** A delivery one day late and one
thirty days late are both `DATE_MISMATCH`. A real reconciliation has a
tolerance window: a slip inside it is not a discrepancy, and treating it as
one buries the slips that matter. Nothing measures this yet, which is why it
sits below rather than being done already - it is a policy question about the
data, not a defect anybody has caught.

**2. Finish measuring what the history compaction costs.** The harness is
built and one round is recorded above: the saving is real, the effect on
accuracy is unknown, because two cases in a single run cannot separate a
difference from noise. What it needs is repetition - the same two
conversations, several runs each way, on the default model - which is a quota
problem rather than a programming one. Until then the compaction stays, on the
strength of the number that was measured rather than the one that was not.

**3. Cache the comparison across runs.** It is recomputed on every invocation.
Fine for 31 rows, wasteful for a real extract; parquet beside the CSVs with a
staleness check would fix it. Last because `chat` already reuses one comparison
for a whole session, so the waste is now per session rather than per question.

## License

MIT - see [LICENSE](LICENSE).
