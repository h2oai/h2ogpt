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
                                   description="Parameters for the action (e.g., search query, URL, text to find)")


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
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": """You are a research agent that uses careful reasoning to find specific information in academic papers.

Before taking any action, analyze the task and current state:
1. Break down the key components of what you're looking for
2. Consider what information you already have and what's missing
3. Plan your strategy for finding the missing information

For arXiv searches:
- Use specific date formats: YYYY-MM or YYYY-MM-DD
- Include site:arxiv.org in searches
- Use category tags when relevant (e.g., physics.soc-ph)
- Use quotes for exact phrases
- Combine search terms strategically

Search progression strategy:
1. Start with specific, targeted searches
2. If not found, broaden the search systematically
3. Consider alternative ways to find the information
4. Track what has been tried and avoid repetition

Review your previous reasoning and actions to avoid repetition and build on what you've learned.
Provide your thoughts in a clear, step-by-step manner before deciding on an action.
Your response must be valid JSON matching the provided schema."""}
        ]

        # Enhanced JSON schema with reasoning structure
        self.action_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "object",
                    "properties": {
                        "task_analysis": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Break down the key components and requirements of the task"
                        },
                        "current_state": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Analyze what we know and don't know from current results"
                        },
                        "strategy": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Plan the next steps to find the information"
                        },
                        "reflection": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Reflect on past attempts and what was learned"
                        }
                    },
                    "required": ["task_analysis", "current_state", "strategy", "reflection"]
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
                        "query": {
                            "type": "object",
                            "properties": {
                                "search_type": {
                                    "type": "string",
                                    "enum": ["arxiv", "general", "specific_date"]
                                },
                                "terms": {"type": "string"},
                                "date": {"type": "string"},
                                "category": {"type": "string"}
                            },
                            "required": ["search_type", "terms"]
                        },
                        "url": {"type": "string"},
                        "text": {"type": "string"},
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "next_steps": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["reasoning", "action", "reason", "params"]
        }

    def _format_history_summary(self) -> str:
        """Format a summary of previous actions and findings"""
        summary = "\nHistory Summary:\n"
        summary += "\nSearch Attempts:\n"
        for search in self.search_attempts:
            summary += f"- {search}\n"

        summary += "\nBrowsing History:\n"
        for visit in self.browsing_history:
            summary += f"- {visit}\n"

        summary += "\nKey Decisions and Findings:\n"
        for message in self.conversation_history:
            if message["role"] == "assistant":
                try:
                    content = json.loads(message["content"])
                    if "reasoning" in content:
                        for strategy in content["reasoning"]["strategy"]:
                            summary += f"- Strategy: {strategy}\n"
                        if "reflection" in content["reasoning"]:
                            for reflection in content["reasoning"]["reflection"]:
                                summary += f"- Learned: {reflection}\n"
                except:
                    continue

        return summary

    def get_next_action(self, task: str, current_page_content: str) -> BrowserAction:
        """Get the next browser action with conversation history"""
        action_json = None
        try:
            # Add the current task and content to the conversation
            history_summary = self._format_history_summary()
            current_prompt = f"{task}\n\nCurrent Page Content:\n{current_page_content}\n{history_summary}"

            self.conversation_history.append({"role": "user", "content": current_prompt})

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
            print("\nReasoning Process:")
            print("\nTask Analysis:")
            for step in action_dict["reasoning"]["task_analysis"]:
                print(f"- {step}")
            print("\nCurrent State:")
            for state in action_dict["reasoning"]["current_state"]:
                print(f"- {state}")
            print("\nStrategy:")
            for strategy in action_dict["reasoning"]["strategy"]:
                print(f"- {strategy}")
            print("\nReflection on Past Attempts:")
            for reflection in action_dict["reasoning"]["reflection"]:
                print(f"- {reflection}")

            if action_dict["action"] == "finish":
                if "params" not in action_dict:
                    action_dict["params"] = {}
                action_dict["params"].update({
                    "key_findings": action_dict["params"].get("key_findings", ["No specific findings discovered"]),
                    "sources": action_dict["params"].get("sources", self.browsing_history),
                    "next_steps": action_dict["params"].get("next_steps", ["Further research needed"])
                })

            return BrowserAction.parse_obj(action_dict)

        except Exception as e:
            traceback.print_exc()
            print(f"Error getting next action: {action_json}\n: {e}")
            error_message = {"role": "assistant", "content": "Error occurred in processing"}
            self.conversation_history.append(error_message)
            return BrowserAction(
                action="finish",
                reason="Error occurred in LLM response",
                params={
                    "key_findings": ["Research incomplete due to error"],
                    "sources": self.browsing_history,
                    "next_steps": ["Retry research with refined prompts"]
                }
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

    def research(self, task: str, max_actions: int = 50) -> ResearchSummary:
        """Conduct research on a given task"""
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
                        sources=action.params.get("sources", self.browsing_history),
                        next_steps=action.params.get("next_steps", ["Further research needed"])
                    )

                current_content = self.execute_action(action)
                time.sleep(2)  # Be nice to servers

        except Exception as e:
            print(f"Error during research: {e}")
            return ResearchSummary(
                key_findings=["Research incomplete due to error"],
                sources=self.browsing_history,
                next_steps=["Retry research with refined prompts"]
            )

        # If we hit max actions, return a summary
        return ResearchSummary(
            key_findings=["Research incomplete - hit maximum number of actions"],
            sources=self.browsing_history,
            next_steps=["Continue research with more allowed actions"]
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


if __name__ == "__main__":
    main()
