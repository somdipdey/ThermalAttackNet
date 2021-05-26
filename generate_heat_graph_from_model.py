def read_csv(filename):
  import csv
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    data_points = []
    for row in csv_reader:
        data_points.append(row[0])
        line_count += 1
    print("Processed {" + str(line_count) + "} lines.")
    return data_points

def generate_heat_plot(data_points, save_path, filename):
    import numpy as np
    import matplotlib.pyplot as plt
    incremental_point = np.linspace(1, len(data_points), len(data_points))
    plt.plot(incremental_point, data_points)
    fig = plt.figure(1)
    plt.savefig(save_path + "/" + filename + '.png')
    plt.close(fig)

def process_files(path, save_path):
    import os
    num_files = 0
    for file in os.listdir(path):
        if ".DS_Store" not in file:
            print("Processesing file: " + path + "/" + file)
            data_points = read_csv(path + "/" + file)
            generate_heat_plot(data_points, save_path, str(num_files+1))
            num_files += 1

    print("Processed {" + str(num_files) + "} files.")

def main():
    process_files("/Source/Path", "/Target/Path")

if __name__ == '__main__':
    main()
