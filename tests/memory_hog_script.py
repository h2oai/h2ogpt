import time


def use_memory():
    # This list will keep growing, consuming more and more memory
    memory_hog = []

    print("Starting memory allocation...")

    # Continuously append large arrays to the list
    while True:
        # Create a large list (about 10 million integers)
        large_list = [i for i in range(10**7)]

        # Append the large list to memory_hog
        memory_hog.append(large_list)

        # Print the current size of the memory_hog list
        print(f"Appended a large list. Current memory_hog length: {len(memory_hog)}")

        # Sleep for 1 second between allocations
        time.sleep(1)


if __name__ == "__main__":
    use_memory()
