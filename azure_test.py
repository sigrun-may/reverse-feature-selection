import sys
import os


def main():
    data_path = sys.argv[1]
    print("data_path: ", data_path)
    # write dummy file to data_path
    save_path = os.path.join(data_path, "experiment_folder")
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "dummy.txt")
    with open(save_file, "w") as f:
        f.write("Hello World!")


if __name__ == "__main__":
    main()
