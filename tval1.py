from tqdm import tqdm
import os
import math
from concurrent.futures import ThreadPoolExecutor
import time

from tvalmetrics import RagScoresCalculator
import tiktoken
from openai import OpenAI

RUN_COMBINE_ESSAYS = True


def get_sorted_essays():
    # Sort the files so that they are uploaded in order
    combined_essays = os.listdir('combined_essays')
    combined_essays.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    print("\n".join(combined_essays))
    return combined_essays


# Iterate through paul_graham_essays folder and upload each file to the assistant
# Apparently you can't upload more than 20 items to openai's rag so let's just combine them and upload those
def combine_essays(group_num):
    if not RUN_COMBINE_ESSAYS:
        return
    essays = os.listdir('paul_graham_essays')
    group_size = math.ceil(len(essays) / group_num)
    print(f"Grouping {len(essays)} files into groups of {group_size} files for a total of {group_num} groups")

    # Create directory for combined essays
    if not os.path.exists('combined_essays'):
        os.makedirs('combined_essays')
    else:
        # delete all files in combined_essays
        for filename in os.listdir('combined_essays'):
            os.remove(f'combined_essays/{filename}')

    group_counter = 0
    file_counter = 1
    curr_file_text = ""
    for filename in tqdm(essays):
        with open('paul_graham_essays/' + filename, 'r') as file:
            curr_file_text += file.read().strip() + "\n\n\n\n"
            group_counter += 1
            if group_counter == group_size:
                # Write to file
                with open(f'combined_essays/paul_graham_{file_counter}.txt', 'w') as combined_file:
                    combined_file.write(curr_file_text)
                    file_counter += 1
                curr_file_text = ""
                group_counter = 0

    if curr_file_text.strip():
        with open(f'combined_essays/paul_graham_{file_counter}.txt', 'w') as combined_file:
            combined_file.write(curr_file_text)
    return get_sorted_essays()


def get_response(prompt, assistant, client):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    max_tries = 180
    try:
        while True:
            if max_tries == 0:
                client.beta.threads.delete(thread.id)
                raise Exception("Max tries exceeded")
            messages = client.beta.threads.messages.list(
                thread_id=thread.id,
            )
            response_message = messages.data[0].content[0].text.value
            if response_message != prompt and response_message.strip():
                annotations = messages.data[0].content[0].text.annotations
                quotes = [x.file_citation.quote for x in annotations if x.file_citation]
                client.beta.threads.delete(thread.id)
                return response_message, quotes
            time.sleep(1)
            max_tries -= 1
    except Exception as e:
        client.beta.threads.delete(thread.id)
        raise e


def load_qa():
    # Load questions from qa_pairs.json
    import json
    with open('qa_pairs.json', 'r') as qa_file:
        qa_pairs = json.load(qa_file)

    question_list = [qa_pair['question'] for qa_pair in qa_pairs]
    print("Questions:\n" + "\n".join(question_list[:4]))

    print()

    answer_list = [qa_pair['answer'] for qa_pair in qa_pairs]
    print("Answers:\n" + "\n".join(question_list[:4]))
    return question_list, answer_list


def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(enc.encode(text))
    return num_tokens


def upload_essays(combined_essays, client):
    # Go through all the combined files and upload them to the assistant
    files = []
    print("Uploading files")
    for filename in tqdm(combined_essays):
        with open('combined_essays/' + filename, 'rb') as essay_file:
            file = client.files.create(
                file=essay_file,
                purpose='assistants'
            )
            files.append(file)
    return files


def create_assistant(files, client):
    return client.beta.assistants.create(
        name=f"OpenAI Rag Test {len(files)} Files",
        instructions=(
            "You are a chatbot that answers questions about Paul Graham's essays. "
            "Use your knowledge base to best respond to questions. "
            "NO MATTER WHAT, DO NOT PULL INFORMATION FROM EXTERNAL KNOWLEDGE. ONLY USE YOUR OWN KNOWLEDGE BASE."
        ),
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id for file in files]
    )


def setup_assistant(file_count, client):
    combined_essays = combine_essays(file_count)
    files = upload_essays(combined_essays, client)
    return create_assistant(files, client)


# Helper function if you want to view the tokens for each file
def show_file_token_count(essays, files):
    # Go through all files in combined_essays and then calculate the tokens for each file
    total_tokens = 0
    count = 0
    end_of_selected_files = False
    for filename in essays:
        with open('combined_essays/' + filename, 'r') as file:
            file_tokens = count_tokens(file.read())
            count += 1
            total_tokens += file_tokens
            if filename not in [file.filename for file in files] and not end_of_selected_files:
                print("--- END OF SELECTED FILES ---")
                end_of_selected_files = True
            print(f"File {count}: {filename} has {file_tokens} tokens. Total tokens: {total_tokens}")


def run_questions(question_list, openai_responses, assistant, openai_context, client):
    # Go through all questions and get responses from openai assistant
    for question in tqdm(question_list[len(openai_responses):]):
        # If there is an exception, try again until we reach 3 tries at max
        max_tries = 3
        while True:
            try:
                openai_response = get_response(question, assistant, client)
                openai_responses.append(openai_response[0])
                openai_context.append(openai_response[1])
                break
            except Exception as e:
                print(e)
                max_tries -= 1
                if max_tries == 0:
                    raise e
                continue


def run_full_test(file_count, question_list, answer_list, score_calculator, client):
    assistant = setup_assistant(file_count, client)
    openai_responses, openai_context = [], []

    def get_openai_response(question):
        max_tries = 10
        while True:
            try:
                response, context = get_response(question, assistant, client)
                return response, context
            except Exception as e:
                print(e)
                max_tries -= 1
                if max_tries == 0:
                    raise e
                continue

    # Using ThreadPoolExecutor to process questions in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Map the process_question function to each question in the list
        results = list(tqdm(executor.map(get_openai_response, question_list), total=len(question_list)))

    openai_responses, openai_context = zip(*results)

    return score_calculator.score_batch(
        question_list=question_list,
        reference_answer_list=answer_list,
        llm_answer_list=openai_responses,
    )


def plot(openai_scores_df):
    import matplotlib.pyplot as plt
    category_counts = openai_scores_df['answer_similarity_score'].value_counts()
    plt.bar(category_counts.index, category_counts.values)

    plt.title('Distribution of scores for 5 documents')
    plt.xlabel('Score')
    plt.ylabel('Count')

    plt.show()


def cleanup(client):
    # Do not run the following unless you want to delete all files and assistants

    def cleanup_files():
        curr_files = list(client.files.list())
        for file in tqdm(curr_files):
            client.files.delete(file.id)

    def cleanup_assistants():
        curr_assistants = list(client.beta.assistants.list())
        for curr_assistant in tqdm(curr_assistants):
            client.beta.assistants.delete(curr_assistant.id)

    def cleanup_all():
        print("Cleaning up files")
        cleanup_files()
        print("Cleaning up assistants")
        cleanup_assistants()

    cleanup_all()


def main():
    model_name = 'gpt-4-1106-preview'
    score_calculator = RagScoresCalculator(
        model=model_name,
        answer_similarity_score=True,
    )
    # client = OpenAI()
    from src.utils import set_openai
    inference_server = ''
    client, async_client, inf_type, deployment_type, base_url, api_version, api_key = \
        set_openai(inference_server, model_name=model_name)

    assistant = setup_assistant(5, client)

    prompt = "What was Airbnb's monthly financial goal to achieve ramen profitability during their time at Y Combinator?"
    openai_response = get_response(prompt, assistant, client)
    print(openai_response)

    question_list, answer_list = load_qa()

    openai_responses, openai_context = [], []
    run_questions(question_list, openai_responses, assistant, openai_context, client)

    openai_batch_scores = score_calculator.score_batch(
        question_list=question_list[:len(openai_responses)],
        reference_answer_list=answer_list[:len(openai_responses)],
        llm_answer_list=openai_responses,
    )

    # If we are doing a single document, the reliability is increased enough that we can multithread
    # openai_batch_scores = run_full_test(1, question_list, answer_list, score_calculator, client)

    openai_scores_df = openai_batch_scores.to_dataframe()
    # Remove overall_score column since we are only using one stat
    openai_scores_df = openai_scores_df.drop(columns=['overall_score'])
    openai_scores_df.describe()
    plot(openai_scores_df)
    # cleanup(client)


if __name__ == '__main__':
    main()
