#include <omp.h>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <iostream>

const int OUTPUT_DATA_DECIMAL_PRECISION = 6;
const int OUTPUT_TIME_DECIMAL_PRECISION = 5;
using matrix = std::vector<std::vector<double>>;
namespace fs = std::filesystem;

struct DataloggerEntry {
    long double Tp;
    long double Ts;
    int process_count;
    bool is_schedule_static;
    int chunk_size;

};

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
    oss << std::fixed << std::setprecision(OUTPUT_DATA_DECIMAL_PRECISION);
    oss.str(std::string());
    std::copy(row.begin(), row.end() - 1, std::ostream_iterator<double>(oss, ";"));
    std::copy(row.end() - 1, row.end(), std::ostream_iterator<double>(oss));
    output << oss.str() << std::endl;
}

matrix &first_stage(matrix &c, bool paralelize) {
    int n = c.size();
    int r, i, j;
    for (r = 0; r < n - 1; r++) {
#pragma omp parallel for shared(c, n, paralelize, r) private(i, j) default(none) if(paralelize)
        for (i = r + 1; i < n; i++) {
            for (j = r + 1; j < n + 1; j++) {
                c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
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

std::vector<double> gauss(matrix c, bool parallelize) {

    printf("Calculating first stage\n");
    c = first_stage(c, parallelize);

    printf("Calculating second stage\n");
    std::vector<double> x = second_stage(c);
    return x;

}


int main(int argc, char *argv[]) {
    printf("Provide input file: ");
    std::string input_file;
    std::cin >> input_file;

    fs::path input_path(fs::absolute(input_file));
    if (!fs::exists(input_path)) {
        printf("Provided file: %s doesnt exists, terminating.\n", input_path.string().c_str());
        exit(1);
    }
    printf("Parsing input csv file\n");
    matrix c = parse_csv(input_path);
    printf("Calculating\n");
    printf("Creating copy of the original matrix\n");
    matrix copy_of_original;
    copy_of_original.resize(c.size());
    for (std::size_t i = 0; i < copy_of_original.size(); i++) {
        copy_of_original[i] = c[i];
    }
    printf("finished creating copy\n");
    double start_time = omp_get_wtime();
    std::vector<double> x_parallelized = gauss(c, true);
    double end_time = omp_get_wtime();
    double Tp = end_time - start_time;
    printf("Parallized it took: %f seconds\n", Tp);
    start_time = omp_get_wtime();
    std::vector<double> x_sequential = gauss(copy_of_original, false);
    end_time = omp_get_wtime();
    double Ts = end_time - start_time;
    printf("Sequential it took: %f seconds\n", Ts);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(OUTPUT_TIME_DECIMAL_PRECISION);
    oss << "X_" << Ts << "_" << Tp << ".csv";
    printf("Saving output\n");
    save_matrix(x_parallelized, oss.str());
    return 0;
}