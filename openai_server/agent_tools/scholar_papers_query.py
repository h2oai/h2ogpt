import os
import argparse
import requests
import json
from semanticscholar import SemanticScholar
import arxiv


def setup_argparse():
    parser = argparse.ArgumentParser(description="Academic Paper Search Utility")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Number of results to return")
    parser.add_argument("-f", "--fields", nargs='+',
                        default=['title', 'authors', 'venue', 'year', 'abstract', 'citationCount',
                                 'influentialCitationCount', 'openAccessPdf', 'tldr', 'references', 'externalIds'],
                        help="Fields to include in the results (Semantic Scholar only)")
    parser.add_argument("-s", "--sort", choices=['relevance', 'citations'], default='relevance',
                        help="Sort order for results (Semantic Scholar only)")
    parser.add_argument("-y", "--year", type=int, nargs=2, metavar=('START', 'END'),
                        help="Year range for papers (e.g., -y 2000 2023)")
    parser.add_argument("-a", "--author", type=str, help="Filter by author name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full abstracts")
    parser.add_argument("-d", "--download", action="store_true", help="Attempt to download PDFs")
    parser.add_argument("-o", "--output_dir", type=str, default="papers", help="Output directory for downloaded PDFs")
    parser.add_argument("--output", type=str, default="papers", help="Output file name for JSON file")
    parser.add_argument("-j", "--json", action="store_true", help="Output results as JSON")
    parser.add_argument("-r", "--references", type=int, default=0,
                        help="Number of references to include (Semantic Scholar only)")
    parser.add_argument("--source", choices=['semanticscholar', 'arxiv'], default='semanticscholar',
                        help="Choose the source for paper search (default: semanticscholar)")
    return parser.parse_args()


def search_papers_semanticscholar(sch, args):
    search_kwargs = {
        'query': args.query,
        'limit': args.limit,
        'fields': args.fields,
        'sort': args.sort
    }
    if args.year:
        search_kwargs['year'] = f"{args.year[0]}-{args.year[1]}"
    if args.author:
        search_kwargs['author'] = args.author
    return sch.search_paper(**search_kwargs)


def search_papers_arxiv(args):
    search = arxiv.Search(
        query=args.query,
        max_results=args.limit,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    return list(search.results())


def print_paper_info_semanticscholar(paper, index, args):
    info = {
        "index": index,
        "title": paper.title,
        "authors": ', '.join([author.name for author in paper.authors]) if paper.authors else 'N/A',
        "venue": paper.venue,
        "year": paper.year,
        "citations": paper.citationCount,
        "influential_citations": paper.influentialCitationCount,
        "externalIds": paper.externalIds,
    }
    if paper.abstract:
        info["abstract"] = paper.abstract if args.verbose else (
            paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract)
    if paper.openAccessPdf:
        info["open_access_pdf"] = {
            "url": paper.openAccessPdf['url'],
            "status": paper.openAccessPdf['status']
        }
    if hasattr(paper, 'tldr') and paper.tldr:
        info["tldr"] = paper.tldr.text
    if args.references > 0 and hasattr(paper, 'references'):
        info["references"] = [ref.title for ref in paper.references[:args.references]]

    print_info(info, args)


def print_paper_info_arxiv(paper, index, args):
    info = {
        "index": index,
        "title": paper.title,
        "authors": ', '.join(author.name for author in paper.authors),
        "year": paper.published.year,
        "abstract": paper.summary if args.verbose else (
            paper.summary[:200] + "..." if len(paper.summary) > 200 else paper.summary),
        "arxiv_url": paper.entry_id,
        "pdf_url": paper.pdf_url,
    }
    print_info(info, args)


def print_info(info, args):
    if args.json:
        print(json.dumps(info, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(info, f, indent=2)
    else:
        for key, value in info.items():
            if key == "open_access_pdf":
                print(f"   Open Access PDF: {value['url']} (Status: {value['status']})")
            elif key == "references":
                print(f"   Top {len(value)} References:")
                for ref in value:
                    print(f"     - {ref}")
            else:
                print(f"   {key.capitalize()}: {value}")
        print("-" * 50)


def download_pdf_semanticscholar(paper, output_dir):
    if paper.openAccessPdf and paper.openAccessPdf['url']:
        pdf_url = paper.openAccessPdf['url']
        filename = f"{output_dir}/{paper.paperId}.pdf"
        download_pdf(pdf_url, filename)
    else:
        print("   No open access PDF available for download")


def download_pdf_arxiv(paper, output_dir):
    pdf_url = paper.pdf_url
    filename = f"{output_dir}/{paper.get_short_id()}.pdf"
    download_pdf(pdf_url, filename)


def download_pdf(pdf_url, filename):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"   PDF downloaded: {filename}")
    except requests.RequestException as e:
        print(f"   Failed to download PDF: {e}")


def main():
    args = setup_argparse()

    if args.source == 'semanticscholar':
        api_key = os.environ.get("S2_API_KEY")
        if not api_key:
            print("Warning: S2_API_KEY environment variable not set. Some features may be limited.")
        sch = SemanticScholar(api_key=api_key)
        papers = search_papers_semanticscholar(sch, args)
        print_func = print_paper_info_semanticscholar
        download_func = download_pdf_semanticscholar
    else:  # arxiv
        papers = search_papers_arxiv(args)
        print_func = print_paper_info_arxiv
        download_func = download_pdf_arxiv

    if not args.json:
        print(f"Top {args.limit} papers for query '{args.query}' from {args.source}:")
        print("-" * 50)

    if args.download:
        os.makedirs(args.output_dir, exist_ok=True)

    for i, paper in enumerate(papers, 1):
        print_func(paper, i, args)
        if args.download:
            download_func(paper, args.output_dir)
        if i == args.limit:
            break

    print("""\n\nRemember to not only use these scientific scholar paper listings,
but also use ask_question_about_documents.py to ask questions about URLs or PDF documents,
ask_question_about_image.py to ask questions about images,
or download_web_video.py to download videos, etc.
A general google or bing search might be advisable if no good results are present here or PDFs of interest are not available.
If you have not found a good response to the user's original query, continue to write executable code to do so.
""")

if __name__ == "__main__":
    main()
