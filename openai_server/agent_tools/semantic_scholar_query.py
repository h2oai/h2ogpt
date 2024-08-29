import os
import argparse
import requests
import json
from semanticscholar import SemanticScholar


def setup_argparse():
    parser = argparse.ArgumentParser(description="Semantic Scholar Search Utility")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Number of results to return")
    parser.add_argument("-f", "--fields", nargs='+',
                        default=['title', 'authors', 'venue', 'year', 'abstract', 'citationCount',
                                 'influentialCitationCount', 'openAccessPdf', 'tldr', 'references', 'externalIds'],
                        help="Fields to include in the results")
    parser.add_argument("-s", "--sort", choices=['relevance', 'citations'], default='relevance',
                        help="Sort order for results")
    parser.add_argument("-y", "--year", type=int, nargs=2, metavar=('START', 'END'),
                        help="Year range for papers (e.g., -y 2000 2023)")
    parser.add_argument("-a", "--author", type=str, help="Filter by author name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full abstracts")
    parser.add_argument("-d", "--download", action="store_true", help="Attempt to download PDFs")
    parser.add_argument("-o", "--output", type=str, default="papers", help="Output directory for downloaded PDFs")
    parser.add_argument("-j", "--json", action="store_true", help="Output results as JSON")
    parser.add_argument("-r", "--references", type=int, default=0, help="Number of references to include")
    return parser.parse_args()


def search_papers(sch, args):
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


def print_paper_info(paper, index, args):
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

    if hasattr(paper, 'externalIds') and paper.externalIds:
        info["externalIds"] = paper.externalIds

    if args.references > 0 and hasattr(paper, 'references'):
        info["references"] = [ref.title for ref in paper.references[:args.references]]

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key, value in info.items():
            if key == "open_access_pdf":
                print(f"   Open Access PDF: {value['url']} (Status: {value['status']})")
            elif key == "references":
                print(f"   Top {len(value)} References:")
                for ref in value:
                    print(f"     - {ref}")
            elif key == "abstract":
                print(f"   Abstract: {value}")
            elif key == "tldr":
                print(f"   TLDR: {value}")
            else:
                print(f"   {key.capitalize()}: {value}")
        print("-" * 50)


def download_pdf(paper, output_dir):
    if paper.openAccessPdf and paper.openAccessPdf['url']:
        pdf_url = paper.openAccessPdf['url']
        filename = f"{output_dir}/{paper.paperId}.pdf"
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"   PDF downloaded: {filename}")
        except requests.RequestException as e:
            print(f"   Failed to download PDF: {e}")
    else:
        print("   No open access PDF available for download")


def main():
    args = setup_argparse()

    api_key = os.environ.get("S2_API_KEY")
    if not api_key:
        print("Warning: S2_API_KEY environment variable not set. Some features may be limited.")

    sch = SemanticScholar(api_key=api_key)

    papers = search_papers(sch, args)

    if not args.json:
        print(f"Top {args.limit} papers for query '{args.query}':")
        print("-" * 50)

    if args.download:
        os.makedirs(args.output, exist_ok=True)

    for i, paper in enumerate(papers, 1):
        print_paper_info(paper, i, args)
        if args.download:
            download_pdf(paper, args.output)
        if i == args.limit:
            break


if __name__ == "__main__":
    main()
