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
        self.found_info: Dict[str, Any] = {}

        # Enhanced system prompt for better search strategy
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": """You are a research agent that finds academic papers using a systematic approach.

Research Strategy Rules:
1. Break down complex queries into simpler sub-queries
2. Start with broad searches and progressively narrow down
3. Focus on definitive identifiers first (dates, authors, exact titles)
4. Only mention specific details (like figures) when examining full papers
5. Never include figure details in initial searches
6. Verify paper dates exactly - don't settle for "close enough"
7. For arXiv papers:
   - Initial search should focus on submission date and broad topic
   - Use arXiv ID patterns (YYMM.NNNNN) when found
   - Remember abstracts won't contain figure details
8. When examining papers:
   - Check submission date first
   - Only proceed to detailed analysis if date matches
   - Look for sections likely to contain figures (Results, Discussion)
9. Track what's been confirmed vs what still needs verification
10. Don't combine separate search criteria in one query unless necessary

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
                            "enum": [
                                "initial_search",          # Broad search with key identifiers
                                "date_verification",       # Verify exact dates of papers
                                "paper_identification",    # Find specific papers
                                "detailed_analysis",       # Analyze full paper content
                                "figure_analysis",         # Analyze specific figures
                                "cross_reference",         # Compare information between papers
                                "final_verification"       # Verify all requirements are met
                            ]
                        },
                        "confirmed_facts": {
                            "type": "object",
                            "description": "Facts that have been verified"
                        },
                        "pending_verification": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Facts that still need verification"
                        }
                    },
                    "required": ["task_phase", "confirmed_facts", "pending_verification"]
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

    def get_next_action(self, task: str, current_page_content: str) -> BrowserAction:
        """Get next action with improved search strategy"""
        try:
            # Break down the task into components
            task_analysis = f"""Task Components:
{task}

Current State:
- Page Content: {current_page_content[:500]}...
- Current URL: {self.browsing_history[-1] if self.browsing_history else 'None'}
- Search History: {json.dumps(self.search_attempts[-3:], indent=2)}  # Show last 3 searches
- Confirmed Facts: {json.dumps(self.found_info, indent=2)}
"""
            self.conversation_history.append({"role": "user", "content": task_analysis})

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

            action_json = json.loads(response.choices[0].message.content)
            self.conversation_history.append({"role": "assistant", "content": json.dumps(action_json)})

            # Print reasoning process
            print("\nPhase:", action_json["reasoning"]["task_phase"])
            print("\nConfirmed Facts:")
            for fact, value in action_json["reasoning"]["confirmed_facts"].items():
                print(f"- {fact}: {value}")
            print("\nPending Verification:")
            for item in action_json["reasoning"]["pending_verification"]:
                print(f"- {item}")

            # Update confirmed facts
            if action_json["reasoning"]["confirmed_facts"]:
                self.found_info.update(action_json["reasoning"]["confirmed_facts"])

            # Clean up search queries
            if action_json["action"] == "search":
                query = action_json["params"]["query"]
                if "arxiv.org" not in query and not query.startswith("site:"):
                    query = f"site:arxiv.org {query}"
                action_json["params"]["query"] = query

            return BrowserAction(
                action=action_json["action"],
                reason=action_json["reason"],
                params=action_json["params"]
            )

        except Exception as e:
            traceback.print_exc()
            return BrowserAction(
                action="finish",
                reason=f"Error occurred: {str(e)}",
                params={"error": str(e)}
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
                print(f"Params: {action.params}")
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
