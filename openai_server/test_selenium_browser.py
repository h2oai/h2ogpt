import os
from typing import List, Dict, Any
import json
from datetime import datetime
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from openai_server.agent_tools.selenium_browser import SeleniumBrowser  # From previous artifact

class BrowserAction(BaseModel):
    """Model for browser actions the LLM can take"""
    action: str = Field(..., description="The browser action to take: 'search', 'visit', 'scroll_down', 'scroll_up', 'find_text', 'find_next', 'finish'")
    reason: str = Field(..., description="Reasoning for taking this action")
    params: Dict[str, Any] = Field(default_dict={}, description="Parameters for the action (e.g., search query, URL, text to find)")

class ResearchSummary(BaseModel):
    """Model for the final research summary"""
    key_findings: List[str] = Field(..., description="List of key findings from the research")
    sources: List[str] = Field(..., description="List of sources consulted")
    next_steps: List[str] = Field(..., description="Suggested next steps or areas for further research")

def create_system_prompt() -> str:
    return """You are a research agent that controls a web browser to find information about topics.
You can use these actions:
- 'search': Perform a Bing search (params: {"query": "search terms"})
- 'visit': Visit a specific URL (params: {"url": "https://..."})
- 'scroll_down': Scroll down the page
- 'scroll_up': Scroll up the page
- 'find_text': Search for text on current page (params: {"text": "search terms"})
- 'find_next': Find next occurrence of the current search text
- 'finish': Complete research and provide summary

For each action:
1. Think carefully about what information you need
2. Choose the most appropriate action to find that information
3. Explain your reasoning
4. Be specific with search terms and URLs

You should:
- Follow leads that seem promising
- Verify information across multiple sources
- Take note of important details
- Be strategic about page navigation
- Use find_text to locate specific information on long pages

When you finish:
1. Summarize key findings
2. List sources consulted
3. Suggest next steps

Respond in valid JSON format matching the BrowserAction model.
"""

def create_user_prompt(task: str, current_page_content: str, browsing_history: List[str]) -> str:
    return f"""Research Task: {task}

Current Page Content:
{current_page_content}

Browsing History:
{json.dumps(browsing_history, indent=2)}

Based on this information, what browser action should be taken next? Respond with a valid JSON object including 'action', 'reason', and 'params'.

If you have gathered enough information to complete the task, use the 'finish' action and provide a comprehensive summary.
"""

class ResearchAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str, bing_api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.browser = SeleniumBrowser(
            bing_api_key=bing_api_key,
            downloads_folder="downloads",
            headless=True
        )
        self.browsing_history: List[str] = []

    def get_next_action(self, task: str, current_page_content: str) -> BrowserAction:
        """Get the next browser action from the LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": create_system_prompt()},
                    {"role": "user", "content": create_user_prompt(task, current_page_content, self.browsing_history)}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            action_json = response.choices[0].message.content
            return BrowserAction.parse_raw(action_json)
        except Exception as e:
            print(f"Error getting next action: {e}")
            # Return a safe default action
            return BrowserAction(
                action="finish",
                reason="Error occurred in LLM response",
                params={}
            )

    def execute_action(self, action: BrowserAction) -> str:
        """Execute a browser action and return the resulting page content"""
        try:
            if action.action == "search":
                query = action.params.get("query", "")
                self.browser.visit_page(f"bing:{query}")
                self.browsing_history.append(f"Searched for: {query}")

            elif action.action == "visit":
                url = action.params.get("url", "")
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
                self.browser.find_on_page(text)
                self.browsing_history.append(f"Searched page for: {text}")

            elif action.action == "find_next":
                self.browser.find_next()
                self.browsing_history.append("Moved to next search result")

            return self.browser.viewport

        except Exception as e:
            print(f"Error executing action: {e}")
            return f"Error: {str(e)}"

    def research(self, task: str, max_actions: int = 10) -> ResearchSummary:
        """Conduct research on a given task"""
        print(f"\nStarting research on: {task}")
        current_content = "No content yet"

        for i in range(max_actions):
            print(f"\nStep {i + 1}")

            action = self.get_next_action(task, current_content)
            print(f"Action: {action.action}")
            print(f"Reason: {action.reason}")

            if action.action == "finish":
                summary = ResearchSummary.parse_raw(json.dumps(action.params))
                return summary

            current_content = self.execute_action(action)
            time.sleep(2)  # Be nice to servers

        # If we hit max actions, return a summary of what we found
        return ResearchSummary(
            key_findings=["Research incomplete - hit maximum number of actions"],
            sources=self.browsing_history,
            next_steps=["Continue research with more allowed actions"]
        )

    def close(self):
        """Clean up resources"""
        self.browser.close()

def main():
    # Load API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME")
    bing_api_key = os.getenv("BING_API_KEY")

    if not openai_api_key or not bing_api_key:
        raise ValueError("Please set OPENAI_API_KEY and BING_API_KEY environment variables")

    # Create and run the agent
    agent = ResearchAgent(openai_api_key, openai_base_url, openai_model_name, bing_api_key)

    try:
        # Example research task
        task = "Research the current state of quantum computing. Focus on recent breakthroughs and major companies involved."

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

if __name__ == "__main__":
    main()