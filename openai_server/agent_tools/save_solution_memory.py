import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Save new solution to memory")
    parser.add_argument("--task", type=str, required=True, help="Task explanation that lead to the error and the solution")
    parser.add_argument("--error", type=str, required=True, help="Error message that was encountered")
    parser.add_argument("--solution", type=str, required=True, help="Solution to the error")
    args = parser.parse_args()

    # Save the solution to memory.
    # Memory is the csv file that is located at './openai_server/agent_tools/solutions_memory.csv'
    # The csv file has the following columns: task, error, solution
    # If the file does not exist, create an empty pandas dataframe with the columns and write it to the file
    # If it exists, append the new solution to the file
    import os
    # print current directory
    print(os.getcwd())
    memory_file = 'solutions_memory.csv'
    if not os.path.exists(memory_file):
        memory = pd.DataFrame(columns=['task', 'error', 'solution'])
        memory.to_csv(memory_file, index=False)
    # read the csv file
    memory = pd.read_csv(memory_file)
    # concat the new solution to the memory
    memory = pd.concat([memory, pd.DataFrame([[args.task, args.error, args.solution]], columns=['task', 'error', 'solution'])], ignore_index=True)
    # write the memory back to the file
    memory.to_csv(memory_file, index=False)
    print(f"New solution for the error has been saved to memory")

if __name__ == "__main__":
    main()
