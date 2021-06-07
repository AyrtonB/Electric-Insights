[![goodtables.io](https://goodtables.io/badge/github/AyrtonB/EI.svg)](https://goodtables.io/github/AyrtonB/EI)

# Electric Insights

[Electric Insights](https://electricinsights.co.uk/) provides a site to "Take a closer look at the supply, demand, price and environmental impact of Britain’s electricity", it also exposes an API which this repository wraps with a Python client. The raw data made available from the Electric Insights site is collated within a Frictionless Data Package, with automated scripts to ensure it remains up-to-date. It is hoped that this repository provides a useful external facing representation of the data that will reduce the number of queries made directly to the Electric Insights API. A number of exploratory notebooks are also included showing some of the analysis this data can be used for.

```bash
python -m electricinsights.retrieval
```