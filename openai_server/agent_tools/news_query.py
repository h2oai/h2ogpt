import requests
import os
import argparse
from datetime import datetime, timedelta


def fetch_everything(api_key, query, sources, from_date, to_date, sort_by, language, page_size):
    base_url = 'https://newsapi.org/v2/everything'

    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'sortBy': sort_by,
        'language': language,
        'pageSize': page_size,
        'apiKey': api_key
    }
    if sources:
        params['sources'] = sources

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()


def fetch_top_headlines(api_key, sources, country, category, page_size):
    base_url = 'https://newsapi.org/v2/top-headlines'

    params = {
        'pageSize': page_size,
        'apiKey': api_key
    }
    if sources:
        params['sources'] = sources
    elif country:
        params['country'] = country
        if category:
            params['category'] = category

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()


def display_articles(articles):
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']['name']}")
        print(f"Author: {article.get('author', 'Not specified')}")
        print(f"Published: {article['publishedAt']}")
        print(f"Description: {article.get('description', 'Not available')}")
        print(f"URL: {article['url']}")


def main():
    parser = argparse.ArgumentParser(description="Fetch news articles or top headlines from News API.")
    parser.add_argument("--mode", choices=['everything', 'top-headlines'], default='everything',
                        help="Choose between 'everything' or 'top-headlines' mode. Default is 'everything'.")

    # Common arguments
    parser.add_argument("--sources",
                        help="Comma-separated list of news sources or blogs (e.g., bbc-news,techcrunch,engadget)")
    parser.add_argument("-n", "--num_articles", type=int, default=10,
                        help="Number of articles to retrieve (max 100). Default is 10.")

    # Arguments for 'everything' mode
    parser.add_argument("-q", "--query",
                        help="The search query for news articles (required for 'everything' mode if sources not specified)")
    parser.add_argument("-f", "--from_date", help="The start date for articles (YYYY-MM-DD). Default is 30 days ago.")
    parser.add_argument("-t", "--to_date", help="The end date for articles (YYYY-MM-DD). Default is today.")
    parser.add_argument("-s", "--sort_by", choices=['relevancy', 'popularity', 'publishedAt'],
                        default='publishedAt', help="The order to sort articles in. Default is publishedAt.")
    parser.add_argument("-l", "--language", default='en',
                        help="The 2-letter ISO-639-1 code of the language. Default is 'en'.")

    # Arguments for 'top-headlines' mode
    parser.add_argument("-c", "--country",
                        help="The 2-letter ISO 3166-1 code of the country. Default is 'us' if sources not specified.")
    parser.add_argument("--category",
                        choices=['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology'],
                        help="The category for top headlines. Optional.")

    args = parser.parse_args()

    # Ensure num_articles is within the allowed range
    args.num_articles = max(1, min(args.num_articles, 100))

    # Get API key from environment variable
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        parser.error("NEWS_API_KEY environment variable is not set")

    try:
        if args.mode == 'everything':
            if not args.query and not args.sources:
                parser.error("Either --query or --sources is required for 'everything' mode")

            # Set default dates if not provided
            today = datetime.now().date()
            from_date = args.from_date or (today - timedelta(days=30)).isoformat()
            to_date = args.to_date or today.isoformat()

            result = fetch_everything(api_key, args.query, args.sources, from_date, to_date, args.sort_by,
                                      args.language, args.num_articles)

            print(f"\nMode: Everything")
            if args.query:
                print(f"Query: '{args.query}'")
            if args.sources:
                print(f"Sources: {args.sources}")
            print(f"From: {from_date} To: {to_date}")
            print(f"Sort by: {args.sort_by}")
            print(f"Language: {args.language}")
        else:  # top-headlines mode
            if not args.sources and not args.country:
                args.country = 'us'  # Default to 'us' if neither sources nor country specified
            result = fetch_top_headlines(api_key, args.sources, args.country, args.category, args.num_articles)

            print(f"\nMode: Top Headlines")
            if args.sources:
                print(f"Sources: {args.sources}")
            elif args.country:
                print(f"Country: {args.country}")
                if args.category:
                    print(f"Category: {args.category}")

        print(f"\nRequested articles: {args.num_articles}")
        print(f"Total results available: {result['totalResults']}")
        print(f"Articles retrieved: {len(result['articles'])}")

        if result['articles']:
            display_articles(result['articles'])
        else:
            print("No articles found.")
    except requests.RequestException as e:
        print(f"An error occurred while fetching news: {e}")

    print("""\n\nRemember to not only use these news snippets,
but also use ask_question_about_documents.py to ask questions about URLs or documents,
ask_question_about_image.py to ask questions about images,
or download_web_video.py to download videos, etc.
If you have not found a good response to the user's original query, continue to write executable code to do so.
""")


if __name__ == "__main__":
    main()
