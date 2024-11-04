import os
import time
import traceback
import json
import requests
import diskcache as dc
from typing import Optional, Union, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse, quote
from dataclasses import dataclass, field

from openai import OpenAI
# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementNotInteractableException,
    WebDriverException,
    StaleElementReferenceException,
)
from selenium.webdriver.common.keys import Keys

# Pydantic imports
from pydantic import BaseModel, Field


# Define the SeleniumBrowser class
class SeleniumBrowser:
    """A Selenium-based web browser that implements all SimpleTextBrowser functionality plus human-like interactions."""

    def __init__(
            self,
            start_page: Optional[str] = None,
            viewport_size: Optional[int] = 1024 * 8 * 4,
            downloads_folder: Optional[Union[str, None]] = None,
            bing_api_key: Optional[Union[str, None]] = None,
            request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
            headless: bool = True,
            timeout: int = 10
    ):
        self.start_page = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size
        self.downloads_folder = downloads_folder
        self.bing_api_key = bing_api_key
        self.request_kwargs = request_kwargs
        self.timeout = timeout
        self.history: List[Tuple[str, float]] = []
        self.page_title: Optional[str] = None
        self.bing_cache = dc.Cache(".cache/bing")

        # Initialize Selenium WebDriver
        self.options = webdriver.ChromeOptions()
        if headless:
            self.options.add_argument("--headless")
            self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option("useAutomationExtension", False)

        if downloads_folder:
            self.options.add_experimental_option(
                "prefs", {
                    "download.default_directory": os.path.abspath(downloads_folder),
                    "download.prompt_for_download": False,
                }
            )

        self.driver = webdriver.Chrome(options=self.options)
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        # Set initial viewport size
        self.driver.set_window_size(1024, self.viewport_size // 16)  # Approximate pixels from characters

        # Initialize search state
        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_matches: List[Any] = []
        self._find_on_page_current_index: int = -1

        # Visit start page
        self.set_address(self.start_page)
        # Add new attributes for handling UI interactions
        self.wait = WebDriverWait(self.driver, timeout)
        self.current_frame = None

    def wait_for_page_load(self, timeout: int = 10):
        """Wait for the page to load completely."""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script('return document.readyState') == 'complete'
            )
        except TimeoutException:
            print("Page did not load completely within the timeout period.")

    def find_element_by_xpath(self, xpath: str, timeout: int = 5) -> Optional[Any]:
        """Find an element using XPath with timeout and error handling."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            return element
        except (TimeoutException, NoSuchElementException, StaleElementReferenceException, WebDriverException) as e:
            print(f"Exception in find_element_by_xpath: {e}")
            return None

    def find_elements_by_xpath(self, xpath: str, timeout: int = 5) -> List[Any]:
        """Find multiple elements using XPath with timeout and error handling."""
        try:
            elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located((By.XPATH, xpath))
            )
            return elements
        except (TimeoutException, NoSuchElementException, StaleElementReferenceException, WebDriverException) as e:
            print(f"Exception in find_elements_by_xpath: {e}")
            return []

    def click(self, selector: str, timeout: int = 5) -> bool:
        """Click an element using various selector strategies."""
        try:
            # Try different selector strategies
            for by, value in [
                (By.XPATH, selector),
                (By.CSS_SELECTOR, selector),
                (By.ID, selector),
                (By.LINK_TEXT, selector)
            ]:
                try:
                    element = WebDriverWait(self.driver, timeout).until(
                        EC.element_to_be_clickable((by, value))
                    )
                    if element:
                        # Scroll element into view
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(0.5)  # Allow time for scroll
                        # Try direct click first
                        try:
                            element.click()
                            return True
                        except ElementNotInteractableException:
                            # If direct click fails, try JavaScript click
                            self.driver.execute_script("arguments[0].click();", element)
                            return True
                except Exception as e:
                    print(f"Exception during click attempt with {by}={value}: {e}")
                    continue
            print("Element to click not found.")
            return False
        except Exception as e:
            print(f"Click failed: {str(e)}")
            return False

    def fill_form(self, form_data: Dict[str, Any]) -> bool:
        """Fill out a form using provided data."""
        try:
            for field_name, value in form_data.items():
                found = False
                # Find all input elements
                inputs = self.driver.find_elements(By.TAG_NAME, 'input') + self.driver.find_elements(By.TAG_NAME,
                                                                                                     'textarea') + self.driver.find_elements(
                    By.TAG_NAME, 'select')
                for element in inputs:
                    name = element.get_attribute('name') or ''
                    id_attr = element.get_attribute('id') or ''
                    placeholder = element.get_attribute('placeholder') or ''
                    aria_label = element.get_attribute('aria-label') or ''
                    if field_name.lower() in (name.lower(), id_attr.lower(), placeholder.lower(), aria_label.lower()):
                        found = True
                        # Handle different input types
                        input_type = element.get_attribute('type')
                        if input_type == 'checkbox':
                            if value and not element.is_selected():
                                element.click()
                            elif not value and element.is_selected():
                                element.click()
                        elif input_type == 'radio':
                            if str(value).lower() == element.get_attribute('value').lower():
                                element.click()
                        else:
                            # Clear and fill text inputs
                            try:
                                element.clear()
                                element.send_keys(str(value))
                            except ElementNotInteractableException:
                                self.driver.execute_script(
                                    f"arguments[0].value = '{str(value)}';",
                                    element
                                )
                        break
                if not found:
                    print(f"Form field '{field_name}' not found.")
            return True
        except Exception as e:
            print(f"Form fill failed: {str(e)}")
            return False

    def handle_date_filter(self, date_values: Dict[str, str]) -> bool:
        """Handle date filter interfaces."""
        try:
            # Find date inputs
            date_inputs = self.driver.find_elements(By.XPATH, "//input[@type='date']")
            if date_inputs:
                for field, value in date_values.items():
                    for input_element in date_inputs:
                        name = input_element.get_attribute('name') or ''
                        id_attr = input_element.get_attribute('id') or ''
                        placeholder = input_element.get_attribute('placeholder') or ''
                        aria_label = input_element.get_attribute('aria-label') or ''
                        if field.lower() in (name.lower(), id_attr.lower(), placeholder.lower(), aria_label.lower()):
                            input_element.send_keys(value)
                            break
                return True
            else:
                # Try to find any date pickers or inputs
                inputs = self.driver.find_elements(By.TAG_NAME, 'input')
                for field, value in date_values.items():
                    for input_element in inputs:
                        input_type = input_element.get_attribute('type') or ''
                        if input_type.lower() in ['text', 'date']:
                            name = input_element.get_attribute('name') or ''
                            id_attr = input_element.get_attribute('id') or ''
                            placeholder = input_element.get_attribute('placeholder') or ''
                            aria_label = input_element.get_attribute('aria-label') or ''
                            if field.lower() in (
                            name.lower(), id_attr.lower(), placeholder.lower(), aria_label.lower()):
                                input_element.send_keys(value)
                                break
                return True
        except Exception as e:
            print(f"Date filter handling failed: {str(e)}")
            return False

    def handle_category_filter(self, category: str) -> bool:
        """Handle category/subject filter interfaces."""
        try:
            # Find elements that may represent categories
            elements = self.driver.find_elements(By.XPATH, "//*[text()]")
            for element in elements:
                text = element.text.strip().lower()
                if category.lower() in text:
                    # Scroll element into view
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                    time.sleep(0.5)
                    # Try to click the element
                    try:
                        element.click()
                        return True
                    except ElementNotInteractableException:
                        # Try JavaScript click if direct click fails
                        self.driver.execute_script("arguments[0].click();", element)
                        return True
            print(f"Category '{category}' not found.")
            return False
        except Exception as e:
            print(f"Category filter handling failed: {str(e)}")
            return False

    @property
    def address(self) -> str:
        """Return the current page URL."""
        return self.driver.current_url

    def set_address(self, uri_or_path: str) -> None:
        """Navigate to a new address."""
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self.driver.get("about:blank")
        elif uri_or_path.startswith("bing:"):
            self._bing_search(uri_or_path[len("bing:"):].strip())
        else:
            if not any(uri_or_path.startswith(prefix) for prefix in ["http:", "https:", "file:"]):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    self.history[-1] = (uri_or_path, self.history[-1][1])

            try:
                self.driver.get(uri_or_path)
                self.wait_for_page_load(timeout=self.timeout)
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                print(f"Timeout waiting for page to load: {uri_or_path}")
            except Exception as e:
                print(f"Exception during page load: {e}")

        self.page_title = self.driver.title
        self._find_on_page_query = None
        self._find_on_page_matches = []
        self._find_on_page_current_index = -1

    def _extract_visible_text(self) -> str:
        """Extract visible text content from the current page."""
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except Exception as e:
            print(f"Error extracting visible text: {e}")
            return ""

    @property
    def viewport(self) -> str:
        """Return the visible text in the current viewport."""
        return self._extract_visible_text()

    def page_down(self) -> None:
        """Scroll down one viewport."""
        self.driver.execute_script(
            f"window.scrollBy(0, {self.viewport_size // 16});"
        )
        time.sleep(0.5)  # Allow time for content to load

    def page_up(self) -> None:
        """Scroll up one viewport."""
        self.driver.execute_script(
            f"window.scrollBy(0, -{self.viewport_size // 16});"
        )
        time.sleep(0.5)  # Allow time for content to load

    def find_on_page(self, query: str) -> Union[str, None]:
        """Search for text on the current page."""
        if not query:
            return None

        # Reset search state if this is a new query
        if query != self._find_on_page_query:
            self._find_on_page_query = query

            # Use JavaScript to find all matching text nodes
            js_code = """
            function getAllTextNodes() {
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                const nodes = [];
                let node;
                while (node = walker.nextNode()) {
                    nodes.push(node);
                }
                return nodes;
            }

            function findMatches(query) {
                const nodes = getAllTextNodes();
                const matches = [];
                const regex = new RegExp(query, 'gi');

                nodes.forEach((node) => {
                    const text = node.textContent;
                    if (regex.test(text)) {
                        matches.push({
                            text: text,
                            xpath: getXPath(node)
                        });
                    }
                });
                return matches;
            }

            function getXPath(node) {
                let path = '';
                while (node && node.nodeType === Node.TEXT_NODE) {
                    node = node.parentNode;
                }
                while (node && node.nodeType === Node.ELEMENT_NODE) {
                    let count = 1;
                    let sibling = node.previousSibling;
                    while (sibling) {
                        if (sibling.nodeType === Node.ELEMENT_NODE && sibling.nodeName === node.nodeName) {
                            count++;
                        }
                        sibling = sibling.previousSibling;
                    }
                    path = `/${node.nodeName.toLowerCase()}[${count}]${path}`;
                    node = node.parentNode;
                }
                return path;
            }

            return findMatches(arguments[0]);
            """

            try:
                self._find_on_page_matches = self.driver.execute_script(js_code, query)
                self._find_on_page_current_index = -1
            except Exception as e:
                print(f"Error during find_on_page: {e}")
                self._find_on_page_matches = []

        return self.find_next()

    def find_next(self) -> Union[str, None]:
        """Move to the next search match."""
        if not self._find_on_page_matches:
            print("No matches found for the query.")
            return None

        self._find_on_page_current_index = (self._find_on_page_current_index + 1) % len(self._find_on_page_matches)
        match = self._find_on_page_matches[self._find_on_page_current_index]

        # Scroll the match into view
        try:
            element = self.driver.find_element(By.XPATH, match['xpath'])
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)

            # Highlight the match
            self.driver.execute_script("""
                arguments[0].style.backgroundColor = 'yellow';
                setTimeout(() => {
                    arguments[0].style.backgroundColor = '';
                }, 1000);
            """, element)
        except Exception as e:
            print(f"Error highlighting match: {e}")

        return self.viewport

    def _bing_api_call(self, query: str) -> Dict[str, Any]:
        """Make a Bing API search request."""
        if self.bing_cache is not None:
            cached = self.bing_cache.get(query)
            if cached is not None:
                return cached

        if self.bing_api_key is None:
            raise ValueError("Missing Bing API key.")

        request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}

        if "headers" not in request_kwargs:
            request_kwargs["headers"] = {}
        request_kwargs["headers"]["Ocp-Apim-Subscription-Key"] = self.bing_api_key

        if "params" not in request_kwargs:
            request_kwargs["params"] = {}
        request_kwargs["params"].update({
            "q": query,
            "textDecorations": False,
            "textFormat": "raw"
        })

        response = None
        for _ in range(10):
            try:
                response = requests.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    **request_kwargs
                )
                response.raise_for_status()
                break
            except Exception as e:
                print(f"Bing API call failed: {e}")
                time.sleep(1)

        if response is None:
            raise requests.exceptions.RequestException("Failed to fetch Bing search results.")

        results = response.json()

        if self.bing_cache is not None:
            self.bing_cache.set(query, results)

        return results

    def _bing_search(self, query: str) -> None:
        """Perform a Bing search and display results."""
        results = self._bing_api_call(query)

        # Format results as HTML
        html_content = f"""
        <html>
        <head>
            <title>Bing Search Results - {query}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .result {{ margin-bottom: 20px; }}
                .title {{ color: #1a0dab; text-decoration: none; }}
                .url {{ color: #006621; font-size: 0.9em; }}
                .snippet {{ color: #545454; }}
            </style>
        </head>
        <body>
            <h1>Search Results for: {query}</h1>
        """

        if "webPages" in results:
            html_content += "<h2>Web Results</h2>"
            for page in results["webPages"]["value"]:
                html_content += f"""
                <div class="result">
                    <a href="{page['url']}" class="title">{page['name']}</a><br>
                    <span class="url">{page['url']}</span><br>
                    <span class="snippet">{page['snippet']}</span>
                </div>
                """

        if "news" in results:
            html_content += "<h2>News Results</h2>"
            for news in results["news"]["value"]:
                date = news.get("datePublished", "").split("T")[0]
                html_content += f"""
                <div class="result">
                    <a href="{news['url']}" class="title">{news['name']}</a><br>
                    <span class="url">{news['url']}</span><br>
                    <span class="snippet">{news['description']}</span><br>
                    <span class="date">Published: {date}</span>
                </div>
                """

        html_content += "</body></html>"

        # Load the results in the browser
        self.driver.execute_script(f"document.documentElement.innerHTML = arguments[0]", html_content)
        self.page_title = f"Bing Search Results - {query}"

    def visit_page(self, path_or_uri: str) -> str:
        """Visit a page and return its viewport content."""
        self.set_address(path_or_uri)
        return self.viewport

    def download_file(self, url: str) -> str:
        """Download a file and return its location."""
        if not self.downloads_folder:
            raise ValueError("Downloads folder not configured")

        self.driver.get(url)
        time.sleep(2)  # Wait for download to start

        # Get the latest file in the downloads directory
        files = [os.path.join(self.downloads_folder, f) for f in os.listdir(self.downloads_folder)]
        if not files:
            return "No file downloaded"

        latest_file = max(files, key=os.path.getctime)
        return f"File downloaded to: {latest_file}"

    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()

    def __del__(self):
        """Ensure browser is closed on deletion."""
        self.close()


# Define the SearchInterface class
@dataclass
class SearchInterface:
    """Tracks discovered search capabilities of a website"""
    domain: str
    advanced_search_url: Optional[str] = None
    date_filter_elements: List[Dict[str, Any]] = field(default_factory=list)
    category_filter_elements: List[Dict[str, Any]] = field(default_factory=list)
    search_fields: List[Dict[str, Any]] = field(default_factory=list)


# Define the BrowserAction class
class BrowserAction(BaseModel):
    """Model for browser actions the LLM can take"""
    action: str = Field(...,
                        description="The browser action to take: 'search', 'visit', 'click', 'fill_form', "
                                    "'scroll_down', 'scroll_up', 'find_text', 'find_next', 'finish'")
    reason: str = Field(..., description="Reasoning for taking this action")
    plan: Optional[List[str]] = Field(None, description="High-level plan steps.")
    params: Dict[str, Any] = Field(default_factory=dict,
                                   description="Parameters for the action")


# Define the ResearchSummary class
class ResearchSummary(BaseModel):
    """Model for the final research summary"""
    key_findings: List[str] = Field(..., description="List of key findings from the research")
    intermediate_results: Dict[str, Any] = Field(default_factory=dict,
                                                 description="Important intermediate findings")
    sources: List[str] = Field(..., description="List of sources consulted")
    next_steps: List[str] = Field(..., description="Suggested next steps or areas for further research")


# Now define the ResearchAgent class
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

        # Track discovered search interfaces
        self.search_interfaces: Dict[str, SearchInterface] = {}
        self.current_search_interface: Optional[SearchInterface] = None

        # Track browsing history and found information
        self.browsing_history: List[str] = []
        self.search_attempts: List[str] = []
        self.found_info: Dict[str, Any] = {}

        # Enhanced system prompt for search interface discovery
        self.conversation_history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": """You are a research agent that intelligently discovers and uses advanced search capabilities.

**Important Guidelines**:

- **Search Strategy**:
  - Use advanced search features of specific sites when appropriate.
  - Avoid using search operators (like 'site:') that are specific to certain search engines unless you are using that search engine.
  - Do not use Google-specific search operators on other websites.

- **Task Decomposition**:
  - Break down the main task into high-level subtasks.
  - Plan your approach before executing actions.
  - Only include specific details (like figure captions) when you have accessed the relevant documents.

- **Action Planning**:
  - Start with locating relevant resources broadly.
  - Narrow down to specifics after identifying potential sources.
  - Document each step and reason for your actions.

**Your response must be valid JSON matching the provided schema. Do not include any extra information or commentary outside the JSON response.**"""
            }
        ]

        self.action_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Detailed reasoning about the action."
                },
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "High-level plan steps."
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "search",  # Perform a search
                        "visit",  # Visit a URL
                        "click",  # Click an element
                        "fill_form",  # Fill out a form
                        "scroll_down",  # Scroll down
                        "scroll_up",  # Scroll up
                        "find_text",  # Find text on page
                        "find_next",  # Find next occurrence
                        "finish"  # Complete task
                    ]
                },
                "params": {"type": "object"}
            },
            "required": ["reasoning", "plan", "action", "params"]
        }

    def discover_search_interface(self, domain: str) -> SearchInterface:
        """Discover advanced search capabilities for a domain"""
        if domain in self.search_interfaces:
            return self.search_interfaces[domain]

        interface = SearchInterface(domain=domain)

        # Extract elements from the page
        page_elements = self._extract_page_elements()

        # Use LLM to interpret elements
        interpretation = self._interpret_page_elements(page_elements)

        # Update interface based on interpretation
        if interpretation:
            interface.advanced_search_url = interpretation.get('advanced_search_url')
            interface.date_filter_elements = interpretation.get('date_filter_elements', [])
            interface.category_filter_elements = interpretation.get('category_filter_elements', [])
            interface.search_fields = interpretation.get('search_fields', [])

        self.search_interfaces[domain] = interface
        return interface

    def _extract_page_elements(self) -> List[Dict[str, Any]]:
        """Extract relevant elements from the current page"""
        elements_info = []

        # Collect all links, buttons, inputs, selects
        elements = self.browser.driver.find_elements(By.XPATH, "//a | //button | //input | //select")

        for element in elements:
            try:
                tag_name = element.tag_name
                text = element.text.strip()
                attributes = element.get_property('attributes')
                attrs = {attr['name']: attr['value'] for attr in attributes}
                elements_info.append({
                    'tag': tag_name,
                    'text': text,
                    'attributes': attrs
                })
            except Exception as e:
                print(f"Error extracting element info: {e}")
                continue

        return elements_info

    def _interpret_page_elements(self, elements_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use the LLM to interpret the meaning of page elements"""
        try:
            # Prepare the prompt for the LLM
            prompt = f"""Given the following elements on a webpage, identify the advanced search link URL, date filter elements, category filter elements, and search fields.
The elements are:\n\n{json.dumps(elements_info, indent=2)}
Provide the information in the following JSON format:"""
            interp_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "advanced_search_url": {
                        "type": ["string", "null"],
                        "format": "uri",
                        "description": "URL for advanced search or null if not provided."
                    },
                    "date_filter_elements": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Identifiers for date filter elements."
                        },
                        "description": "List of element identifiers for date filters."
                    },
                    "category_filter_elements": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Identifiers for category filter elements."
                        },
                        "description": "List of element identifiers for category filters."
                    },
                    "search_fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Identifiers for search field elements."
                        },
                        "description": "List of element identifiers for search fields."
                    }
                },
                "required": ["date_filter_elements", "category_filter_elements", "search_fields"],
                "additionalProperties": False
            }

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[dict(role='user', content=prompt)],
                temperature=0.0,
                max_tokens=1500,
                extra_body=dict(
                    guided_json=interp_schema,
                    enable_caching=self.enable_caching,
                    response_format=dict(type="json_object"),
                )
            )

            interpretation_json = json.loads(response.choices[0].message.content.strip())
            return interpretation_json
        except Exception as e:
            print(f"Error interpreting page elements: {e}")
            return {}

    def get_next_action(self, task: str, current_page_content: str) -> BrowserAction:
        """Get next action with dynamic search interface discovery"""
        try:
            # Get current domain
            current_url = self.browser.address
            domain = urlparse(current_url).netloc if current_url else None

            # Update search interface knowledge
            if domain and domain not in self.search_interfaces:
                self.discover_search_interface(domain)

            # Include search interface information in context
            context = {
                "task": task,
                "current_content": current_page_content,
                "current_url": current_url,
                "browsing_history": self.browsing_history,
                "search_interfaces": {
                    domain: {
                        'advanced_search_url': self.search_interfaces[domain].advanced_search_url,
                        'date_filter_elements': self.search_interfaces[domain].date_filter_elements,
                        'category_filter_elements': self.search_interfaces[domain].category_filter_elements,
                        'search_fields': self.search_interfaces[domain].search_fields,
                    }
                    for domain in self.search_interfaces
                }
            }

            # Prepare the planning prompt
            planning_prompt = f"""
Given the task: "{task}", and the current context, break down the task into high-level subtasks before deciding on the next action.

Current URL: {current_url}
Browsing History: {self.browsing_history}

Provide a JSON object with the following format:
{{
  "reasoning": "Your reasoning here",
  "plan": ["First subtask", "Second subtask", "..."],
  "action": "next action to take",
  "params": {{}}
}}

Remember to avoid using search operators specific to certain search engines on other sites.

Your response must be valid JSON matching the schema provided.
"""

            # Combine context and planning prompt into a single message
            user_message_content = f"{json.dumps(context)}\n\n{planning_prompt}"

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message_content})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=0.0,
                max_tokens=1500,
                extra_body=dict(
                    guided_json=self.action_schema,
                    enable_caching=self.enable_caching,
                    response_format=dict(type="json_object"),
                )
            )

            action_json = json.loads(response.choices[0].message.content.strip())
            self.conversation_history.append({"role": "assistant", "content": json.dumps(action_json)})

            # Process the action before returning
            if action_json["action"] == "search":
                # Update search attempts tracking
                query = action_json["params"].get("query", "")
                if query and query not in self.search_attempts:
                    self.search_attempts.append(query)

            return BrowserAction(
                action=action_json["action"],
                reason=action_json["reasoning"],
                plan=action_json.get("plan", []),
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
        """Execute a browser action using discovered search interfaces"""
        try:
            current_url = self.browser.address
            domain = urlparse(current_url).netloc if current_url else None
            interface = self.search_interfaces.get(domain) if domain else None

            if action.action == "search":
                # If no current page, start with a known search engine or arXiv
                if not domain:
                    if "arxiv" in action.params.get("query", "").lower():
                        self.browser.visit_page("https://arxiv.org")
                    else:
                        self.browser.visit_page("https://www.bing.com")
                    current_url = self.browser.address
                    domain = urlparse(current_url).netloc
                    interface = self.search_interfaces.get(domain)

                # Now proceed with search
                if interface and interface.advanced_search_url:
                    self.browser.visit_page(interface.advanced_search_url)
                    self._configure_advanced_search(interface, action.params)
                else:
                    # Attempt to discover search interface dynamically
                    self.discover_search_interface(domain)
                    interface = self.search_interfaces.get(domain)
                    if interface and interface.advanced_search_url:
                        self.browser.visit_page(interface.advanced_search_url)
                        self._configure_advanced_search(interface, action.params)
                    else:
                        # Fall back to basic search
                        self._execute_basic_search(action.params)

            elif action.action == "click":
                selector = action.params.get("selector", "")
                if selector:
                    success = self.browser.click(selector)
                    if not success:
                        print(f"Failed to click element with selector: {selector}")
                else:
                    print("No selector provided for click action.")

            elif action.action == "fill_form":
                form_data = action.params.get("form_data", {})
                if form_data:
                    self.browser.fill_form(form_data)
                else:
                    print("No form data provided for fill_form action.")

            elif action.action == "visit":
                url = action.params.get("url", "")
                if url:
                    new_domain = urlparse(url).netloc
                    if new_domain and new_domain not in self.search_interfaces:
                        self.discover_search_interface(new_domain)
                    self.browser.visit_page(url)
                    self.browsing_history.append(f"Visited: {url}")
                else:
                    print("No URL provided for visit action.")

            elif action.action in ["scroll_down", "scroll_up", "find_text", "find_next"]:
                if action.action == "scroll_down":
                    self.browser.page_down()
                    self.browsing_history.append("Scrolled down")
                elif action.action == "scroll_up":
                    self.browser.page_up()
                    self.browsing_history.append("Scrolled up")
                elif action.action == "find_text":
                    text = action.params.get("text", "")
                    if text:
                        result = self.browser.find_on_page(text)
                        self.browsing_history.append(f"Searched page for: {text}")
                    else:
                        print("No text provided for find_text action.")
                elif action.action == "find_next":
                    result = self.browser.find_next()
                    self.browsing_history.append("Moved to next search result")

            return self.browser.viewport

        except Exception as e:
            print(f"Error executing action: {e}")
            return f"Error: {str(e)}"

    def _configure_advanced_search(self, interface: SearchInterface, params: Dict[str, Any]):
        """Configure advanced search using discovered interface"""
        try:
            # Handle date parameters
            if 'date' in params:
                date_param = params['date']
                if isinstance(date_param, dict):
                    self.browser.handle_date_filter(date_param)

            # Handle category/subject parameters
            if 'category' in params:
                category = params['category']
                self.browser.handle_category_filter(category)

            # Handle search fields
            if 'query' in params:
                query = params['query']
                self.browser.fill_form({'query': query})

            # Submit the search
            # Try to find a submit button
            submit_button = None
            buttons = self.browser.driver.find_elements(By.TAG_NAME, 'button')
            for button in buttons:
                if 'submit' in (button.get_attribute('type') or '').lower() or 'search' in (button.text or '').lower():
                    submit_button = button
                    break
            if submit_button:
                submit_button.click()
            else:
                # Try to press Enter key in search input
                search_input = self.browser.driver.find_element(By.TAG_NAME, 'input')
                if search_input:
                    search_input.send_keys(Keys.RETURN)
                else:
                    print("No submit mechanism found.")

        except Exception as e:
            print(f"Error configuring advanced search: {e}")
            # Fall back to basic search
            self._execute_basic_search(params)

    def _execute_basic_search(self, params: Dict[str, Any]):
        """Execute basic search when advanced interface unavailable"""
        query = params.get('query', '')
        if not query:
            print("No query provided for basic search.")
            return

        # Sanitize the query to remove inappropriate search operators
        query = self._sanitize_query(query)

        try:
            # Look for basic search input
            search_input = None
            inputs = self.browser.driver.find_elements(By.TAG_NAME, 'input')
            for input_element in inputs:
                input_type = input_element.get_attribute('type') or ''
                name = input_element.get_attribute('name') or ''
                placeholder = input_element.get_attribute('placeholder') or ''
                if input_type.lower() in ['search', 'text'] and 'search' in (name.lower() + placeholder.lower()):
                    search_input = input_element
                    break

            if search_input:
                search_input.clear()
                search_input.send_keys(query)
                search_input.send_keys(Keys.RETURN)
                self.browsing_history.append(f"Searched for: {query}")
            else:
                print("Search input not found; attempting URL-based search.")
                # Fall back to URL-based search if needed
                self.browser.visit_page(f"?q={quote(query)}")
                self.browsing_history.append(f"Searched for: {query}")
        except Exception as e:
            print(f"Error executing basic search: {e}")
            # Fall back to URL-based search if needed
            self.browser.visit_page(f"?q={quote(query)}")
            self.browsing_history.append(f"Searched for: {query}")

    def _sanitize_query(self, query: str) -> str:
        """Remove search operators that are not applicable to the current search engine"""
        # For non-Google search engines, remove 'site:' operator and similar
        disallowed_operators = ['site:', 'inurl:', 'intitle:', 'intext:']
        for op in disallowed_operators:
            query = query.replace(op, '')
        return query.strip()

    def research(self, task: str, max_actions: int = 50) -> ResearchSummary:
        """Conduct research using discovered search interfaces"""
        print(f"\nStarting research on: {task}")
        current_content = "No content yet"
        plan = []

        try:
            for i in range(max_actions):
                print(f"\nStep {i + 1}")

                action = self.get_next_action(task, current_content)
                print(f"Action: {action.action}")
                print(f"Params: {action.params}")
                print(f"Reason: {action.reason}")

                # Update the plan if provided
                if action.plan:
                    plan = action.plan
                    print(f"Updated Plan: {plan}")

                if action.action == "finish":
                    return ResearchSummary(
                        key_findings=action.params.get("key_findings", []),
                        intermediate_results=self.found_info,
                        sources=self.browsing_history,
                        next_steps=action.params.get("next_steps", [])
                    )

                current_content = self.execute_action(action)

                # Allow time for dynamic content to load
                time.sleep(2)

                # Update interface knowledge based on results
                if action.action == "search":
                    current_url = self.browser.address
                    domain = urlparse(current_url).netloc
                    if domain:
                        self.discover_search_interface(domain)

            return ResearchSummary(
                key_findings=["Research incomplete - hit maximum actions"],
                intermediate_results=self.found_info,
                sources=self.browsing_history,
                next_steps=["Continue with more actions allowed"]
            )

        except Exception as e:
            print(f"Error during research: {e}")
            return ResearchSummary(
                key_findings=["Research incomplete due to error"],
                intermediate_results=self.found_info,
                sources=self.browsing_history,
                next_steps=["Retry with simplified approach"]
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
