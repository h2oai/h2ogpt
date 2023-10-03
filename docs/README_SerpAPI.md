## h2oGPT integration with LangChain and SerpAPI

Web search augments LLM context with additional information obtained from duck duck go (can be changed in code) search results.

* Install search package
```bash
pip install -r reqs_optional/requirements_optional_agents.txt
````

* Setup account at https://serpapi.com/ (they have some number of free searches for free accounts)

* Setup ENV that defines: `SERPAPI_API_KEY`

* Start h2oGPT as normal

* You should see web search available in `Resources`

* Additionally, the SEARCH agent will appear in `Resources` under `Agents`.  These agents are highly experimental and works best with OpenAI at moment.

## Issues

When web search is enabled, it has been seen that eventually it leads to some closing of sys.stdout and one gets these errors:
```
ValueError: I/O operation on closed file.
```
