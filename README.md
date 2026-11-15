free-model-aggregator-02

One-file Python 3.12+ server for the Make.fun trial. It reads the embedded model and prompt documents from MAKEFUN_PROMPT_FILE, registers the exact free-model catalog, and supports OpenAI Chat Completions and Anthropic Messages clients. Anthropic requests are translated to Chat Completions for the trial gateway and translated back, including tools and streaming events.

Routing policy: filter to free models whose declared tools, vision, JSON, and context capabilities satisfy the matching manifest row, then rank by rating_milli descending with manifest order as the stable tie-break. On an upstream HTTP refusal, try the next eligible model and expose up to eight refusals in x-makefun-upstream-failures. Every gateway request carries a max_tokens cap no greater than 8192. The server uses fixed catalog metadata and no clock or random inputs, and exits when the env reports work done.

Run `python3 main.py --list` to print the catalog, or start it in a trial environment with `python3 main.py`.
