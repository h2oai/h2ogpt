## h2oGPT integration with LangChain Agents

Various agents from LangChain are included:
* Search -- Works sometimes with non-OpenAI models after improvements beyond LangChain
* Collection -- Pre-alpha tested
* Python -- Pre-alpha tested, only currently allowed with OpenAI
* CSV -- Works well with OpenAI due to use of Function Tools
* Pandas -- Disabled until load csv/json with pandas.
* JSON -- Alpha tested, only currently allowed with OpenAI
* AutoGPT -- Alpha tested
  * Tools:
    * Search
    * Wikipedia
    * Shell
    * File
    * Python
    * Requests
    * Wolfram Alpha
  * Memory
