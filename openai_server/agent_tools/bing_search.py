import os
import argparse
import json
from azure.core.credentials import AzureKeyCredential
from web_search_client import WebSearchClient
from image_search_client import ImageSearchClient
from news_search_client import NewsSearchClient
from video_search_client import VideoSearchClient

BING_API_KEY = os.environ.get("BING_API_KEY")
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0"


# Example web query:
# python openai_server/agent_tools/bing_search_tool.py -q "Tom Riddle" -t web -l 5 -m en-US -f Week -s Moderate -v -j
# Example image query:
# python openai_server/agent_tools/bing_search.py -q "Mount Fuji" -t image -l 3 -m en-US -s Moderate -v -j
# Example news query:
# python openai_server/agent_tools/bing_search.py -q "artificial intelligence" -t news -l 3 -m en-US -f Day -v -j
# Example video query:
# python openai_server/agent_tools/bing_search.py -q "SpaceX launch" -t video -l 3

def setup_argparse():
    parser = argparse.ArgumentParser(description="Bing Search Utility")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument("-t", "--type", choices=['web', 'image', 'news', 'video'], default='web', help="Type of search")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Number of results to return")
    parser.add_argument("-m", "--market", type=str, default="en-US", help="Market for search results")
    parser.add_argument("-f", "--freshness", choices=[None, 'Day', 'Week', 'Month'], default=None,
                        help="Freshness of results")
    parser.add_argument("-s", "--safe", choices=['Off', 'Moderate', 'Strict'], default='Off',
                        help="Safe search setting")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="Print full descriptions/content")
    parser.add_argument("-j", "--json", action="store_true", default=True, help="Output results as JSON")
    parser.add_argument("--output", type=str, default='', help="Name of file to output JSON result to if set")
    return parser.parse_args()


def search_web(client, args):
    web_data = client.web.search(
        query=args.query,
        count=args.limit,
        market=args.market,
        freshness=args.freshness,
        safe_search=args.safe
    )
    return web_data.web_pages.value if web_data.web_pages else []


def search_images(client, args):
    image_results = client.images.search(
        query=args.query,
        count=args.limit,
        market=args.market,
        freshness=args.freshness,
        safe_search=args.safe
    )
    return image_results.value if image_results else []


def search_news(client, args):
    news_result = client.news.search(
        query=args.query,
        count=args.limit,
        market=args.market,
        freshness=args.freshness,
        safe_search=args.safe
    )
    return news_result.value if news_result else []


def search_videos(client, args):
    video_result = client.videos.search(
        query=args.query,
        count=args.limit,
        market=args.market,
        freshness=args.freshness,
        safe_search=args.safe
    )
    return video_result.value if video_result else []


def print_web_result(result, args):
    info = {
        "name": result.name,
        "url": result.url,
        "snippet": result.snippet if args.verbose else (
            result.snippet[:200] + "..." if len(result.snippet) > 200 else result.snippet)
    }
    print_info(info, args)


def print_image_result(result, args):
    info = {
        "name": result.name,
        "content_url": result.content_url,
        "thumbnail_url": result.thumbnail_url,
        "host_page_url": getattr(result, 'host_page_url', 'N/A')
    }
    print_info(info, args)


def print_news_result(result, args):
    info = {
        "name": result.name,
        "url": result.url,
        "description": result.description if args.verbose else (
            result.description[:200] + "..." if len(result.description) > 200 else result.description),
        "date_published": result.date_published,
        "provider": result.provider[0].name if result.provider else "Unknown"
    }
    print_info(info, args)


def print_video_result(result, args):
    info = {
        "name": result.name,
        "content_url": result.content_url,
        "thumbnail_url": getattr(result, 'thumbnail_url', 'N/A'),
        "duration": getattr(result, 'duration', 'N/A'),
        "creator": result.creator.name if getattr(result, 'creator', None) else "Unknown"
    }
    print_info(info, args)


def print_info(info, args):
    if args.json:
        if args.output:
            with open(args.output, 'wt') as f:
                json.dump(info, f, indent=2, default=str)
            print(f"\nJSON output saved to: {args.output}")
        else:
            print("\nJSON output:")
            print(json.dumps(info, indent=2, default=str))
    else:
        for key, value in info.items():
            print(f"   {key.capitalize()}: {value}")
        print("-" * 50)


def bing_search():
    args = setup_argparse()

    if not BING_API_KEY:
        raise ValueError("BING_API_KEY environment variable is not set.")

    credential = AzureKeyCredential(BING_API_KEY)

    if args.type == 'web':
        client = WebSearchClient(endpoint=BING_ENDPOINT, credential=credential)
        results = search_web(client, args)
        print_func = print_web_result
    elif args.type == 'image':
        client = ImageSearchClient(endpoint=BING_ENDPOINT, credential=credential)
        results = search_images(client, args)
        print_func = print_image_result
    elif args.type == 'news':
        client = NewsSearchClient(endpoint=BING_ENDPOINT, credential=credential)
        results = search_news(client, args)
        print_func = print_news_result
    elif args.type == 'video':
        client = VideoSearchClient(endpoint=BING_ENDPOINT, credential=credential)
        results = search_videos(client, args)
        print_func = print_video_result
    else:
        raise ValueError(f"Invalid search type: {args.type}")

    if not args.json:
        print(f"Top {args.limit} {args.type} results for query '{args.query}':")
        print("-" * 50)

    for result in results[:args.limit]:
        print_func(result, args)

    print("""\n\nRemember web snippets are short and often non-specific.
For specific information, you must use ask_question_about_documents.py on URLs or documents,
ask_question_about_image.py for images,
or download_web_video.py for videos, etc.
If you have not found a good response to the user's original query, continue to write executable code to do so.
""")


if __name__ == "__main__":
    bing_search()
