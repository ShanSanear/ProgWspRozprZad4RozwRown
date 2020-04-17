#include <omp.h>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iterator>
#include "argument_parsing.cpp"

const int DECIMAL_PRECISION = 6;
using matrix = std::vector<std::vector<double>>;
namespace fs = std::filesystem;

matrix parse_csv(const fs::path &input_csv_file) {
    std::ifstream data(input_csv_file);
    std::string line;
    // Skipping required dimension lines
    std::getline(data, line);
    matrix parsed_csv;
    while (std::getline(data, line)) {
        std::vector<double> parsedRow;
        std::stringstream s(line);
        std::string cell;
        while (std::getline(s, cell, ';')) {
            parsedRow.push_back(std::stod(cell));
        }

        parsed_csv.push_back(parsedRow);
    }
    return parsed_csv;
};

void save_matrix(const std::vector<double> &row, const std::string &output_file_path) {
    std::ofstream output(output_file_path);

    output << row.size() << std::endl;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(DECIMAL_PRECISION);
    oss.str(std::string());
    std::copy(row.begin(), row.end() - 1, std::ostream_iterator<double>(oss, ";"));
    std::copy(row.end() - 1, row.end(), std::ostream_iterator<double>(oss));
    output << oss.str() << std::endl;
}

matrix &first_stage(matrix &c) {
    double val;
    int n = c.size();
    for (int r = 0; r < n - 1; r++) {
        for (int i = r + 1; i < n; i++) {
            for (int j = r + 1; j < n + 1; j++) {
                val = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
                c[i][j] = val;
            }
        }
    }
    return c;
}

std::vector<double> second_stage(const matrix &c) {
    int n = c.size();
    std::vector<double> x(n);
    double s;
    x[n - 1] = c[n - 1][n] / c[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
        s = 0;
        for (int r = i; r < n; r++) {
            s = s + c[i][r] * x[r];
        }
        x[i] = (c[i][n] - s) / c[i][i];
    }
    return x;
}

void gauss(matrix c) {
    c = first_stage(c);

    std::vector<double> x = second_stage(c);
    save_matrix(x, "sol4Cpp.csv");
}


int main(int argc, char *argv[]) {
    auto result = parse_arguments(argc, argv);
    fs::path input_path(fs::absolute("input4.csv"));
    matrix input_matrix = parse_csv(input_path);
    gauss(input_matrix);
    return 0;
}