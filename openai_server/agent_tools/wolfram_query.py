import wolframalpha
import requests
import os
import argparse


def sanitize_filename(name):
    bad_chars = ['[', ']', ',', '/', '\\', '\\w', '\\s', '-', '+', '\"', '\'', '>', '<', ' ', '=', ')', '(', ':', '^']
    for char in bad_chars:
        name = name.replace(char, "_")
    return name


def extract_and_save_images(query, app_id, output_dir):
    # Create a client with your app ID
    client = wolframalpha.Client(app_id)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Send the query
    res = client.query(query)

    saved_files = []
    if res['@success']:
        try:
            # Print the result
            print("<basic_results>")
            print(next(res.results).text)
            print("</basic_results>")
        except StopIteration:
            pass

        print("\n\n")
        print("<detailed_results>")
        for i, pod in enumerate(res.pods):
            print(f"\nPod: {pod.title}")
            for j, sub in enumerate(pod.subpods):
                # Print plaintext if available
                if sub.plaintext:
                    print(f"  Subpod {j + 1} Text: {sub.plaintext}")

                # Save image if available
                if hasattr(sub, 'img'):
                    image_url = sub.img.src
                    try:
                        # Download the image
                        response = requests.get(image_url)
                        response.raise_for_status()

                        # Determine the file extension
                        content_type = response.headers.get('content-type')
                        ext = content_type.split('/')[-1] if content_type else 'png'

                        title = sanitize_filename(pod.title)[:20]
                        sub_title = sanitize_filename(sub.img.title.strip())[:20]

                        # Create a filename
                        filename = f"image_{title}_{sub_title}_{i}_{j}.{ext}"
                        filepath = os.path.join(output_dir, filename)

                        # Save the image
                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                        saved_files.append(filepath)
                        print(f"  Saved image: {filepath}")
                    except requests.RequestException as e:
                        print(f"  Error downloading {image_url}: {e}")
        print("</detailed_results>")
    else:
        print(
            "Script ran, but query was not successful. Please try a simpler input (e.g. instead of 'plot rule 30', just say 'rule 30') and try again.")
        print("Error: ", res['@error'])

    return saved_files


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract and save images and text from Wolfram Alpha based on a query.")
    parser.add_argument("-q", "--query", type=str, required=True, help="The query to send to Wolfram Alpha")
    parser.add_argument("-o", "--output_dir", "--file", default="wolfram_images",
                        help="Output directory for saved images (default: wolfram_images)")
    parser.add_argument("-a", "--appid", help="Your Wolfram Alpha App ID")

    # Parse arguments
    args = parser.parse_args()

    # Get App ID from environment variable if not provided as an argument
    app_id = args.appid or os.environ.get("WOLFRAM_ALPHA_APPID")
    if not app_id:
        parser.error(
            "Wolfram Alpha App ID must be provided either as an argument or as WOLFRAM_ALPHA_APP_ID environment variable")

    try:
        print(f"Query: {args.query}\n")
        saved_files = extract_and_save_images(args.query, app_id, args.output_dir)
        print(f"\nSummary: Saved {len(saved_files)} images to {args.output_dir}/")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
