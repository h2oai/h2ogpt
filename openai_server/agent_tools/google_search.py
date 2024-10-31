import os
import argparse
import json
from typing import Dict, Any
from serpapi import (
    SerpApiClient, GoogleSearch, BingSearch, BaiduSearch, YandexSearch,
    YahooSearch, EbaySearch, HomeDepotSearch, YoutubeSearch, GoogleScholarSearch,
    WalmartSearch, AppleAppStoreSearch, NaverSearch
)

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

# Dictionary to translate user-friendly service names to tbm values
GOOGLE_SERVICES = {
    "web": "",
    "image": "isch",
    "local": "lcl",
    "video": "vid",
    "news": "nws",
    "shopping": "shop",
    "patents": "pts",
}

# List of all supported language codes
# https://serpapi.com/google-languages
ALL_LANGUAGE_CODES = [
    "af", "ak", "sq", "ws", "am", "ar", "hy", "az", "eu", "be", "bem", "bn", "bh", "xx-bork", "bs", "br", "bg", "bt",
    "km", "ca", "chr", "ny", "zh-cn", "zh-tw", "co", "hr", "cs", "da", "nl", "xx-elmer", "en", "eo", "et", "ee", "fo",
    "tl", "fi", "fr", "fy", "gaa", "gl", "ka", "de", "el", "kl", "gn", "gu", "xx-hacker", "ht", "ha", "haw", "iw",
    "he", "hi", "hu", "is", "ig", "id", "ia", "ga", "it", "ja", "jw", "kn", "kk", "rw", "rn", "xx-klingon", "kg",
    "ko", "kri", "ku", "ckb", "ky", "lo", "la", "lv", "ln", "lt", "loz", "lg", "ach", "mk", "mg", "ms", "ml", "mt",
    "mv", "mi", "mr", "mfe", "mo", "mn", "sr-me", "my", "ne", "pcm", "nso", "no", "nn", "oc", "or", "om", "ps", "fa",
    "xx-pirate", "pl", "pt", "pt-br", "pt-pt", "pa", "qu", "ro", "rm", "nyn", "ru", "gd", "sr", "sh", "st", "tn",
    "crs", "sn", "sd", "si", "sk", "sl", "so", "es", "es-419", "su", "sw", "sv", "tg", "ta", "tt", "te", "th", "ti",
    "to", "lua", "tum", "tr", "tk", "tw", "ug", "uk", "ur", "uz", "vu", "vi", "cy", "wo", "xh", "yi", "yo", "zu"
]

# Top 10 most commonly used languages (you may want to adjust this list based on your specific use case)
TOP_10_LANGUAGES = [
    ("en", "English"),
    ("es", "Spanish"),
    ("zh-cn", "Chinese (Simplified)"),
    ("ar", "Arabic"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("fr", "French"),
    ("ja", "Japanese"),
    ("ru", "Russian"),
    ("de", "German")
]

# List of all supported country codes
# https://serpapi.com/google-countries
ALL_COUNTRY_CODES = [
    "af", "al", "dz", "as", "ad", "ao", "ai", "aq", "ag", "ar", "am", "aw", "au", "at", "az", "bs", "bh", "bd", "bb",
    "by", "be", "bz", "bj", "bm", "bt", "bo", "ba", "bw", "bv", "br", "io", "bn", "bg", "bf", "bi", "kh", "cm", "ca",
    "cv", "ky", "cf", "td", "cl", "cn", "cx", "cc", "co", "km", "cg", "cd", "ck", "cr", "ci", "hr", "cu", "cy", "cz",
    "dk", "dj", "dm", "do", "ec", "eg", "sv", "gq", "er", "ee", "et", "fk", "fo", "fj", "fi", "fr", "gf", "pf", "tf",
    "ga", "gm", "ge", "de", "gh", "gi", "gr", "gl", "gd", "gp", "gu", "gt", "gn", "gw", "gy", "ht", "hm", "va", "hn",
    "hk", "hu", "is", "in", "id", "ir", "iq", "ie", "il", "it", "jm", "jp", "jo", "kz", "ke", "ki", "kp", "kr", "kw",
    "kg", "la", "lv", "lb", "ls", "lr", "ly", "li", "lt", "lu", "mo", "mk", "mg", "mw", "my", "mv", "ml", "mt", "mh",
    "mq", "mr", "mu", "yt", "mx", "fm", "md", "mc", "mn", "ms", "ma", "mz", "mm", "na", "nr", "np", "nl", "an", "nc",
    "nz", "ni", "ne", "ng", "nu", "nf", "mp", "no", "om", "pk", "pw", "ps", "pa", "pg", "py", "pe", "ph", "pn", "pl",
    "pt", "pr", "qa", "re", "ro", "ru", "rw", "sh", "kn", "lc", "pm", "vc", "ws", "sm", "st", "sa", "sn", "rs", "sc",
    "sl", "sg", "sk", "si", "sb", "so", "za", "gs", "es", "lk", "sd", "sr", "sj", "sz", "se", "ch", "sy", "tw", "tj",
    "tz", "th", "tl", "tg", "tk", "to", "tt", "tn", "tr", "tm", "tc", "tv", "ug", "ua", "ae", "uk", "gb", "us", "um",
    "uy", "uz", "vu", "ve", "vn", "vg", "vi", "wf", "eh", "ye", "zm", "zw"
]

# Top 10 most common countries (you may want to adjust this list based on your specific use case)
TOP_10_COUNTRIES = [
    ("us", "United States"),
    ("gb", "United Kingdom"),
    ("ca", "Canada"),
    ("au", "Australia"),
    ("de", "Germany"),
    ("fr", "France"),
    ("in", "India"),
    ("jp", "Japan"),
    ("br", "Brazil"),
    ("es", "Spain")
]


def setup_argparse():
    parser = argparse.ArgumentParser(description="Multi-Engine Search Utility using SerpApi")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument("-e", "--engine",
                        choices=['google', 'bing', 'baidu', 'yandex', 'yahoo', 'ebay', 'homedepot', 'youtube',
                                 'scholar', 'walmart', 'appstore', 'naver'], default='google',
                        help="Search engine to use")
    parser.add_argument("-l", "--limit", type=int, default=5, help="Number of results to return")
    parser.add_argument("--google_domain", type=str, default="google.com", help="Google domain to use")
    parser.add_argument("--gl", type=str, default="us",
                        help="Country of the search (default: us). Top 10 common countries:\n" +
                             "\n".join(f"  {code}: {name}" for code, name in TOP_10_COUNTRIES) +
                             "\nFor a full list of supported countries, see the documentation.")
    parser.add_argument("--hl", type=str, default="en",
                        help="Language of the search (default: en). Top 10 common languages:\n" +
                             "\n".join(f"  {code}: {name}" for code, name in TOP_10_LANGUAGES) +
                             "\nFor a full list of supported languages, see the documentation.")
    parser.add_argument("--location", type=str, help="Location for the search (optional)")
    parser.add_argument("--type", type=str, default="web",
                        help="Type of Google search to perform. Options:\n"
                             "  web: Regular Google Search (default)\n"
                             "  image: Google Images\n"
                             "  local: Google Local\n"
                             "  video: Google Videos\n"
                             "  news: Google News\n"
                             "  shopping: Google Shopping\n"
                             "  patents: Google Patents\n")
    parser.add_argument("--tbs", type=str, help="Advanced search parameters")
    parser.add_argument("--safe", choices=['active', 'off'], default='off', help="Safe search setting")
    parser.add_argument("--start", type=int, default=0, help="Pagination offset")
    parser.add_argument("--device", choices=['desktop', 'tablet', 'mobile'], default='desktop',
                        help="Device to emulate")
    parser.add_argument("-j", "--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", type=str, default='', help="Name of file to output JSON result to if set")
    parser.add_argument("--keys", nargs='+', help="Specific keys to display in the results")
    return parser.parse_args()


def validate_language(hl: str) -> str:
    if hl not in ALL_LANGUAGE_CODES:
        raise ValueError(f"Invalid language code: {hl}. Please use a valid language code.")
    return hl


def validate_country(gl: str) -> str:
    if gl not in ALL_COUNTRY_CODES:
        raise ValueError(f"Invalid country code: {gl}. Please use a valid country code.")
    return gl


def perform_search(args) -> Dict[str, Any]:
    """
    Perform a search using the specified engine and return the results.
    """
    params = {
        "q": args.query,
        "api_key": SERPAPI_API_KEY,
        "num": max(2, args.limit),
        "device": args.device,
    }

    if args.engine == "google":
        # Translate service to tbm
        tbm = GOOGLE_SERVICES.get(args.type.lower(), "")
        if tbm == 'pts':
            params['num'] = args.limit = min(max(args.limit, 10), 100)
        params.update({
            "google_domain": args.google_domain,
            "gl": validate_country(args.gl),
            "hl": validate_language(args.hl),
            "tbm": tbm,
            "tbs": args.tbs,
            "safe": args.safe,
            "start": args.start,
        })
        if args.location:
            params["location"] = args.location
    elif args.engine in ["bing", "yahoo"]:
        params.update({
            "cc": validate_country(args.gl),
            "setlang": validate_language(args.hl),
        })
    # Add specific parameters for other engines as needed

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    engines = {
        "google": GoogleSearch,
        "bing": BingSearch,
        "baidu": BaiduSearch,
        "yandex": YandexSearch,
        "yahoo": YahooSearch,
        "ebay": EbaySearch,
        "homedepot": HomeDepotSearch,
        "youtube": YoutubeSearch,
        "scholar": GoogleScholarSearch,
        "walmart": WalmartSearch,
        "appstore": AppleAppStoreSearch,
        "naver": NaverSearch,
    }

    search = engines[args.engine](params)
    return search.get_dict()


def save_results_to_file(results: Dict[str, Any], filename: str) -> None:
    """
    Save the full search results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(
        f"""\n# Search results for specific the keys are in this JSON file: {filename}
* One can write python code to extract certain keys from the JSON file, but this file does not contain specific or detailed information for the query, you use should pass specific URLs to ask_question_about_documents.py for specific or detailed information.
""")


def print_results(results: Dict[str, Any], args):
    """
    Print the keys of the search results and a couple of entries for primary results.
    """
    if args.keys:
        print(f"Requested keys for query '{args.query}' using {args.engine} ({args.type} service):")
        for key in args.keys:
            if key in results:
                print(f"\n{key}:")
                print(json.dumps(results[key], indent=2))
            else:
                print(f"\n{key}: Not found in results")
    else:
        print(f"""To extract specific keys, you can repeat the same command and chose the keys you want by using the CLI optional arg: [--keys KEYS [KEYS ...]]
Keys available in the search results for query '{args.query}' using {args.engine} ({args.type} service):
""")

        for key in results.keys():
            print(f"- {key}")

        print("\nSample of primary results:")
        primary_keys = ["organic_results", "news_results", "jobs_results", "shopping_results", "images_results",
                        "video_results", "books_results", "finance_results", "local_results", "patents"]

        for key in primary_keys:
            if key in results and isinstance(results[key], list) and len(results[key]) > 0:
                print(f"\n{key.replace('_', ' ').title()}:")
                for i, result in enumerate(results[key][:args.limit], 1):  # Print first args.limit results
                    if 'title' in result:
                        print(f"  {i}. {result.get('title', '')}:")
                    if 'link' in result:
                        print(f"     URL: {result.get('link', '')}")
                    if 'original' in result:
                        print(f"     original: {result.get('original', '')}")
                    if 'links' in result and 'website' in result['links']:
                        print(f"     Website: {result['links']['website']}")
                    if 'product_link' in result:
                        print(f"     Product Link: {result['product_link']}")
                    if 'snippet' in result:
                        print(f"     Snippet: {result['snippet']}")
                    if 'top_stories' in result:
                        print(f"     Top Stories: {result['top_stories']}")
                break  # Only show sample for the first available primary key

    if args.json:
        if args.output:
            with open(args.output, 'wt') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nFull JSON output saved to: {args.output}")
        else:
            print("\nFull JSON output:")
            print(json.dumps(results, indent=2, default=str))

    print("""\n\nRemember web snippets are short and often non-specific.
For specific information, you must use ask_question_about_documents.py on URLs or documents,
ask_question_about_image.py for images,
or download_web_video.py for videos, etc.
If you have not found a good response to the user's original query, continue to write executable code to do so.
""")


def google_search():
    args = setup_argparse()

    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY environment variable is not set.")

    results = perform_search(args)

    # Print results
    print_results(results, args)

    # Save full results to a file
    save_results_to_file(results, f"{args.engine}_{args.type}_search_results.json")


if __name__ == "__main__":
    google_search()

"""
# Test different search engines
python openai_server/agent_tools/google_search.py -q "artificial intelligence" -e google
python openai_server/agent_tools/google_search.py -q "machine learning" -e bing
python openai_server/agent_tools/google_search.py -q "deep learning" -e baidu
python openai_server/agent_tools/google_search.py -q "neural networks" -e yandex
python openai_server/agent_tools/google_search.py -q "data science" -e yahoo
python openai_server/agent_tools/google_search.py -q "data science" -e scholar

# Test different Google services
python openai_server/agent_tools/google_search.py -q "AI images" -e google --type image
python openai_server/agent_tools/google_search.py -q "AI startups near me" -e google --type local
python openai_server/agent_tools/google_search.py -q "AI tutorials" -e google --type video
python openai_server/agent_tools/google_search.py -q "AI breakthroughs" -e google --type news
python openai_server/agent_tools/google_search.py -q "AI products" -e google --type shopping
python openai_server/agent_tools/google_search.py -q "AI patents" -e google --type patents

# Test with specific keys
python openai_server/agent_tools/google_search.py -q "Python programming" -e google --keys organic_results search_information

# Test with different languages and countries
python openai_server/agent_tools/google_search.py -q "プログラミング" -e google --hl ja --gl jp
python openai_server/agent_tools/google_search.py -q "programmation" -e google --hl fr --gl fr

# Test with JSON output
python openai_server/agent_tools/google_search.py -q "data analysis" -e google -j

# Test pagination
python openai_server/agent_tools/google_search.py -q "machine learning algorithms" -e google --start 10 -n 5

# Test safe search
python openai_server/agent_tools/google_search.py -q "art" -e google --safe active

# Test different devices
python openai_server/agent_tools/google_search.py -q "responsive design" -e google --device mobile
"""
