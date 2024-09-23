import argparse
import pandas as pd
import uuid


def main():
    parser = argparse.ArgumentParser(description="Save new memory to be used for future references")
    parser.add_argument("--task", type=str, required=True, help="Detailed task explanation that lead to the error or the solution")
    parser.add_argument("--error", type=str, required=True, help="Error message that was encountered, if there was any. ")
    parser.add_argument("--solution", type=str, required=True, help="Solution, always includes codes. Full method codes are preferred so that the solution can be recalled as is.")
    args = parser.parse_args()
    # Memory file
    memory_file = f"memory_{str(uuid.uuid4())[:6]}.csv"
    # new memory
    memory = pd.DataFrame([[args.task, args.error, args.solution]], columns=['task', 'error', 'solution'])
    # write the memory back to the file
    memory.to_csv(memory_file, index=False)
    print(f"New memory saved: {memory_file}")

if __name__ == "__main__":
    main()
