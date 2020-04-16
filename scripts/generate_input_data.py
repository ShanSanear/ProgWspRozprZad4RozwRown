import argparse

import numpy as np


def generate_matrix(n, m, lower, higher):
    int_random = np.random.randint(lower, higher, (n, m))
    double_random = np.random.rand(n, m)
    finished_matrix = int_random * double_random
    return finished_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help="N dimension of the output matrix", type=int, default=4)
    parser.add_argument('--output', help="Output file path where the matrix will be placed", default="out.csv")
    parser.add_argument('--solution', help="Output file path where solution will be placed", default="solution.csv")
    parser.add_argument('--lower', help='Low limit of the values possible in matrix', type=float, default=0.0)
    parser.add_argument('--higher', help='High limit of the values possible in matrix', type=float, default=150.0)
    args = parser.parse_args()
    return args


def save_to_csv(path, to_save, size):
    np.savetxt(path, to_save, fmt='%.6f', delimiter=';')
    with open(path, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(f"{size}\n")
        file.write(content)


def main():
    args = parse_args()
    n = args.n
    lower = args.lower
    higher = args.higher
    csv_out_path = args.output
    solution_path = args.solution
    # Example data
    #a = np.array([[2, 1, 4], [3, 2, -1], [-1, 5, 2]])
    #b = np.array([[-1], [2], [-7]])
    a = generate_matrix(n, n, lower, higher)
    b = generate_matrix(n, 1, lower, higher)

    b = b.reshape(-1, 1)
    solution = np.linalg.solve(a, b)
    check = np.allclose(np.dot(a, solution), b)
    if not check:
        raise RuntimeError("Something went horribly wrong")
    csv_out = np.append(a, b, axis=1)

    save_to_csv(csv_out_path, csv_out, size=csv_out.shape[0])
    solution = solution.transpose()
    save_to_csv(solution_path, solution, size=solution.shape[1])


if __name__ == '__main__':
    main()
