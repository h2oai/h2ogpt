from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
from urllib.parse import urljoin
import requests
import time
import os
import diskcache as dc
from typing import Optional, Union, Dict, Any, List, Tuple
from .mdconvert import MarkdownConverter


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
        self._mdconvert = MarkdownConverter()
        self.bing_cache = dc.Cache(".cache/bing")

        # Initialize Selenium WebDriver
        self.options = webdriver.ChromeOptions()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("start-maximized")
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

    def find_element_by_xpath(self, xpath: str, timeout: int = 5) -> Optional[Any]:
        """Find an element using XPath with timeout and error handling."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            return element
        except (TimeoutException, NoSuchElementException):
            return None

    def find_elements_by_xpath(self, xpath: str, timeout: int = 5) -> List[Any]:
        """Find multiple elements using XPath with timeout and error handling."""
        try:
            elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located((By.XPATH, xpath))
            )
            return elements
        except (TimeoutException, NoSuchElementException):
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
                except:
                    continue
            return False
        except Exception as e:
            print(f"Click failed: {str(e)}")
            return False

    def fill_form(self, form_data: Dict[str, Any]) -> bool:
        """Fill out a form using provided data."""
        try:
            for field_name, value in form_data.items():
                # Try various selector strategies to find the form field
                for selector in [
                    f"//input[@name='{field_name}']",
                    f"//input[@id='{field_name}']",
                    f"//textarea[@name='{field_name}']",
                    f"//select[@name='{field_name}']",
                    f"//*[contains(@class, '{field_name}')]"
                ]:
                    element = self.find_element_by_xpath(selector)
                    if element:
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
            return True
        except Exception as e:
            print(f"Form fill failed: {str(e)}")
            return False

    def handle_date_filter(self, filter_type: str, date_values: Dict[str, str]) -> bool:
        """Handle different types of date filter interfaces."""
        try:
            if filter_type == 'form':
                # Handle standard date input forms
                return self.fill_form(date_values)

            elif filter_type == 'dropdown':
                # Handle dropdown date selectors
                for field, value in date_values.items():
                    selector = f"//select[contains(@name, '{field}') or contains(@id, '{field}')]"
                    element = self.find_element_by_xpath(selector)
                    if element:
                        element.send_keys(value)
                return True

            elif filter_type == 'custom':
                # Handle custom date pickers
                date_button = self.find_element_by_xpath(
                    "//button[contains(., 'Date') or contains(@class, 'date')]"
                )
                if date_button:
                    date_button.click()
                    time.sleep(1)  # Wait for picker to appear

                    # Try to find date inputs in the picker
                    for field, value in date_values.items():
                        input_selector = f"//input[contains(@placeholder, '{field}') or contains(@aria-label, '{field}')]"
                        input_element = self.find_element_by_xpath(input_selector)
                        if input_element:
                            input_element.send_keys(value)

                    # Look for apply/done button
                    apply_button = self.find_element_by_xpath(
                        "//button[contains(., 'Apply') or contains(., 'Done')]"
                    )
                    if apply_button:
                        apply_button.click()
                    return True

            return False

        except Exception as e:
            print(f"Date filter handling failed: {str(e)}")
            return False

    def handle_category_filter(self, category: str) -> bool:
        """Handle category/subject filter interfaces."""
        try:
            # Try different category selection patterns
            selectors = [
                f"//input[@type='checkbox'][@value='{category}']",
                f"//input[@type='radio'][@value='{category}']",
                f"//select//option[text()='{category}']",
                f"//a[text()='{category}']",
                f"//button[text()='{category}']",
                f"//*[contains(@class, 'category') and text()='{category}']"
            ]

            for selector in selectors:
                element = self.find_element_by_xpath(selector)
                if element:
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

            return False

        except Exception as e:
            print(f"Category filter handling failed: {str(e)}")
            return False

    @property
    def address(self) -> str:
        """Return the current page URL."""
        return self.history[-1][0] if self.history else self.start_page

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
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                print(f"Timeout waiting for page to load: {uri_or_path}")

        self.page_title = self.driver.title
        self._find_on_page_query = None
        self._find_on_page_matches = []
        self._find_on_page_current_index = -1

    def _extract_visible_text(self) -> str:
        """Extract visible text content from the current page."""
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except:
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

            self._find_on_page_matches = self.driver.execute_script(js_code, query)
            self._find_on_page_current_index = -1

        return self.find_next()

    def find_next(self) -> Union[str, None]:
        """Move to the next search match."""
        if not self._find_on_page_matches:
            return None

        self._find_on_page_current_index = (self._find_on_page_current_index + 1) % len(self._find_on_page_matches)
        match = self._find_on_page_matches[self._find_on_page_current_index]

        # Scroll the match into view
        element = self.driver.find_element(By.XPATH, match['xpath'])
        self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)

        # Highlight the match
        self.driver.execute_script("""
            arguments[0].style.backgroundColor = 'yellow';
            setTimeout(() => {
                arguments[0].style.backgroundColor = '';
            }, 1000);
        """, element)

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
            except:
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
