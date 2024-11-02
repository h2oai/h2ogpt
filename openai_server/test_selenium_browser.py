import os
import traceback
from typing import List, Dict, Any
import json
from datetime import datetime
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from openai_server.agent_tools.selenium_browser import SeleniumBrowser  # From previous artifact


class BrowserAction(BaseModel):
    """Model for browser actions the LLM can take"""
    action: str = Field(...,
                        description="The browser action to take: 'search', 'visit', 'scroll_down', 'scroll_up', 'find_text', 'find_next', 'finish'")
    reason: str = Field(..., description="Reasoning for taking this action")
    params: Dict[str, Any] = Field(default_dict={},
                                   description="Parameters for the action")


class ResearchSummary(BaseModel):
    """Model for the final research summary"""
    key_findings: List[str] = Field(..., description="List of key findings from the research")
    intermediate_results: Dict[str, Any] = Field(default_dict={}, description="Important intermediate findings")
    sources: List[str] = Field(..., description="List of sources consulted")
    next_steps: List[str] = Field(..., description="Suggested next steps or areas for further research")


class ResearchAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str, bing_api_key: str, headless=True):
        self.model_name = model_name
        self.enable_caching = True
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.browser = SeleniumBrowser(
            bing_api_key=bing_api_key,
            downloads_folder="downloads",
            headless=headless
        )
        self.browsing_history: List[str] = []
        self.search_attempts: List[str] = []
        self.found_info: Dict[str, Any] = {}  # Store intermediate findings

        # Enhanced system prompt to encourage discovery-based searching
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": """You are a research agent that specializes in finding academic papers and specific information within them.

Research Strategy:
1. Start with basic searches, but be observant of search features and options on websites
2. If specific dates or criteria are given, look for advanced search options on the current website
3. Learn and use site-specific features as you discover them
4. Think like a human researcher - if a basic search isn't working, look for better search tools
5. Remember useful features you find for future searches
6. Don't be afraid to modify your search strategy based on what you learn

When searching:
- Notice and use advanced search links/options
- Look for date filters and sorting options
- Pay attention to URL patterns and search parameters
- Learn from search result pages and modify approach accordingly

Your response must be valid JSON matching the schema provided."""}
        ]

        self.action_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "object",
                    "properties": {
                        "task_phase": {
                            "type": "string",
                            "enum": ["initial_search", "explore_search_options", "refine_search", "extract_info", "analyze_results"],
                            "description": "Current phase of the research task"
                        },
                        "analysis": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Analysis of current state and needs"
                        },
                        "found_info": {
                            "type": "object",
                            "description": "Information found so far, including discovered search features"
                        }
                    },
                    "required": ["task_phase", "analysis", "found_info"]
                },
                "action": {
                    "type": "string",
                    "enum": ["search", "visit", "scroll_down", "scroll_up", "find_text", "find_next", "finish"]
                },
                "reason": {
                    "type": "string"
                },
                "params": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "url": {"type": "string"},
                        "text": {"type": "string"}
                    }
                }
            },
            "required": ["reasoning", "action", "reason", "params"]
        }

        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": """You are a research agent that specializes in finding academic papers and specific information within them.

Research Strategy:
1. Break complex tasks into sequential phases
2. Complete each phase before moving to the next
3. Store and use intermediate findings
4. Use simple, effective search queries

For arXiv searches:
- Use direct arXiv URLs when possible (e.g., arxiv.org/list/physics.soc-ph/16)
- Start with broad site:arxiv.org searches
- Use CTRL-F (find_text) to locate specific information
- Look for paper IDs and use direct arxiv.org/abs/ID links

Search Construction:
1. Start with simple queries (e.g., "site:arxiv.org AI regulation 2022")
2. Use find_text to locate specific dates or terms
3. When found, visit specific paper pages
4. Use find_text again to locate figures/information

Common arXiv URL patterns:
- Paper listings: arxiv.org/list/[category]/[YYMM]
- Specific papers: arxiv.org/abs/[paper_id]
- Search: arxiv.org/search/?query=[terms]&searchtype=all

Your response must be valid JSON matching the schema provided."""}
        ]

    def get_next_action(self, task: str, current_page_content: str) -> BrowserAction:
        """Get the next browser action with improved search strategy"""
        try:
            # Include found information and current state in the prompt
            context = f"""Task: {task}

Current Page Content:
{current_page_content}

Search History:
{json.dumps(self.search_attempts, indent=2)}

Found Information:
{json.dumps(self.found_info, indent=2)}
"""

            self.conversation_history.append({"role": "user", "content": context})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1500,
                extra_body=dict(
                    guided_json=self.action_schema,
                    enable_caching=self.enable_caching,
                    response_format=dict(type="json_object"),
                )
            )

            action_json = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": action_json})

            if isinstance(action_json, str):
                action_dict = json.loads(action_json)
            else:
                action_dict = action_json

            # Print the reasoning process
            print("\nPhase:", action_dict["reasoning"]["task_phase"])
            print("\nAnalysis:")
            for step in action_dict["reasoning"]["analysis"]:
                print(f"- {step}")

            # Update found information
            if action_dict["reasoning"]["found_info"]:
                self.found_info.update(action_dict["reasoning"]["found_info"])

            if action_dict["action"] == "search":
                # Simplify and clean up search query
                query = action_dict["params"]["query"]
                if "site:arxiv.org" not in query:
                    query = f"site:arxiv.org {query}"
                action_dict["params"]["query"] = query

            return BrowserAction.parse_obj(action_dict)

        except Exception as e:
            traceback.print_exc()
            print(f"Error getting next action: {str(e)}")
            return BrowserAction(
                action="finish",
                reason="Error occurred in LLM response",
                params={
                    "key_findings": ["Research incomplete due to error"],
                    "sources": self.browsing_history,
                    "next_steps": ["Retry with simplified search strategy"]
                }
            )

    def execute_action(self, action: BrowserAction) -> str:
        """Execute a browser action with improved error handling"""
        try:
            if action.action == "search":
                query = action.params.get("query", "")
                if not query:
                    return "Error: No search query provided"

                # Track search attempts
                if query not in self.search_attempts:
                    self.search_attempts.append(query)

                self.browser.visit_page(f"bing:{query}")
                self.browsing_history.append(f"Searched for: {query}")

            elif action.action == "visit":
                url = action.params.get("url", "")
                if not url:
                    return "Error: No URL provided"
                self.browser.visit_page(url)
                self.browsing_history.append(f"Visited: {url}")

            elif action.action == "scroll_down":
                self.browser.page_down()
                self.browsing_history.append("Scrolled down")

            elif action.action == "scroll_up":
                self.browser.page_up()
                self.browsing_history.append("Scrolled up")

            elif action.action == "find_text":
                text = action.params.get("text", "")
                if not text:
                    return "Error: No search text provided"
                result = self.browser.find_on_page(text)
                self.browsing_history.append(f"Searched page for: {text}")
                if result is None:
                    return f"Text not found: {text}"

            elif action.action == "find_next":
                result = self.browser.find_next()
                self.browsing_history.append("Moved to next search result")
                if result is None:
                    return "No more matches found"

            return self.browser.viewport

        except Exception as e:
            print(f"Error executing action: {e}")
            return f"Error: {str(e)}"

    def research(self, task: str, max_actions: int = 50) -> ResearchSummary:
        """Conduct research with improved phase tracking"""
        print(f"\nStarting research on: {task}")
        current_content = "No content yet"

        try:
            for i in range(max_actions):
                print(f"\nStep {i + 1}")

                action = self.get_next_action(task, current_content)
                print(f"Action: {action.action}")
                print(f"Reason: {action.reason}")

                if action.action == "finish":
                    return ResearchSummary(
                        key_findings=action.params.get("key_findings", ["Research incomplete"]),
                        intermediate_results=self.found_info,
                        sources=self.browsing_history,
                        next_steps=action.params.get("next_steps", ["Further research needed"])
                    )

                current_content = self.execute_action(action)
                time.sleep(2)

        except Exception as e:
            print(f"Error during research: {e}")
            return ResearchSummary(
                key_findings=["Research incomplete due to error"],
                intermediate_results=self.found_info,
                sources=self.browsing_history,
                next_steps=["Retry with simplified approach"]
            )

        return ResearchSummary(
            key_findings=["Research incomplete - hit maximum actions"],
            intermediate_results=self.found_info,
            sources=self.browsing_history,
            next_steps=["Continue with more actions allowed"]
        )

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'browser'):
            self.browser.close()

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.close()


def main():
    # Load API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME")
    bing_api_key = os.getenv("BING_API_KEY")

    if not openai_api_key or not bing_api_key:
        raise ValueError("Please set OPENAI_API_KEY and BING_API_KEY environment variables")

    # Create and run the agent
    agent = ResearchAgent(openai_api_key, openai_base_url, openai_model_name, bing_api_key, headless=False)

    try:
        # Example research task
        task = "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"

        summary = agent.research(task)

        print("\nResearch Summary:")
        print("\nKey Findings:")
        for finding in summary.key_findings:
            print(f"- {finding}")

        print("\nSources Consulted:")
        for source in summary.sources:
            print(f"- {source}")

        print("\nSuggested Next Steps:")
        for step in summary.next_steps:
            print(f"- {step}")

    finally:
        agent.close()


def test_selenium_browser():
    main()


if __name__ == "__main__":
    main()
