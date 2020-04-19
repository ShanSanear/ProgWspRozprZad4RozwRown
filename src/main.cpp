#include <ctime>
#include <omp.h>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <iostream>
#include <chrono>

const int OUTPUT_DATA_DECIMAL_PRECISION = 6;
const int OUTPUT_TIME_DECIMAL_PRECISION = 5;
using matrix = std::vector<std::vector<double>>;
namespace fs = std::filesystem;

namespace datalogger {
    struct entry {
        std::string start_execution;
        std::string input_file;
        int equation_count{};
        double Ts{};
        double Tp{};
        int process_count{};
        int chunk_size{};
    };

    std::ostream &operator<<(std::ostream &os, entry const &arg) {
        os << std::setprecision(OUTPUT_TIME_DECIMAL_PRECISION);
        os << arg.start_execution << ";" << arg.input_file << ";" << arg.equation_count << ";"
           << arg.Ts << ";" << arg.Tp << arg.process_count << ";" << arg.chunk_size;
        return os;
    }

    std::string to_string(entry const &arg) {
        std::ostringstream ss;
        ss << arg;
        return std::move(ss).str();
    }
}

matrix load_csv(const fs::path &input_csv_file) {
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
}

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

matrix &first_stage(matrix &c, bool parallelize, int processors_count, int chunk_size) {
    int n = c.size();
    int r, i, j;
    if (!parallelize) {
        printf("First stage sequentially\n");
        for (r = 0; r < n - 1; r++) {
            for (i = r + 1; i < n; i++) {
                for (j = r + 1; j < n + 1; j++) {
                    c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
                }
            }
        }
    } else {
        if (chunk_size != 0) {
            printf("First stage using static schedule with %d processors and %d chunks\n",
                   processors_count, chunk_size);
            for (r = 0; r < n - 1; r++) {
#pragma omp parallel for ordered shared(c, n, parallelize, r) num_threads(processors_count) private(i, j) default(none) \
        schedule(static)
                for (i = r + 1; i < n; i++) {
                    for (j = r + 1; j < n + 1; j++) {
                        c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
                    }
                }
            }
        } else {
            printf("First stage using static schedule with %d processors\n", processors_count);
            for (r = 0; r < n - 1; r++) {
#pragma omp parallel for ordered shared(c, n, parallelize, chunk_size, r) num_threads(processors_count) private( i, j) \
        default(none) schedule(static, chunk_size)
                for (i = r + 1; i < n; i++) {
                    for (j = r + 1; j < n + 1; j++) {
                        c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
                    }
                }
            }
        }
    }
    return c;
}


std::vector<double>
second_stage(const matrix &c, bool parallelize, int processors_count, int chunk_size) {
    int n = c.size();
    std::vector<double> x(n);
    double s = 0.0;
    int i, r;
    x[n - 1] = c[n - 1][n] / c[n - 1][n - 1];
    if (!parallelize) {
        printf("Second stage sequentially\n");
        for (i = n - 2; i >= 0; i--) {
            s = 0;
            for (r = i; r < n; r++) {
                s = s + c[i][r] * x[r];
            }
            x[i] = (c[i][n] - s) / c[i][i];
        }
    } else {
        if (chunk_size != 0) {
            printf("Second stage using static schedule with %d processors and %d chunks\n",
                   processors_count, chunk_size);
            for (i = n - 2; i >= 0; i--) {
                s = 0.0;
#pragma omp parallel for shared(c, n, parallelize, x, i) num_threads(processors_count) private(r) default(none) \
        schedule(static) reduction(+ : s)
                for (r = i; r < n; r++) {
                    s += c[i][r] * x[r];
                }
                x[i] = (c[i][n] - s) / c[i][i];
            }
        } else {
            printf("Second stage using static schedule with %d processors\n", processors_count);

            for (i = n - 2; i >= 0; i--) {
                s = 0.0;
#pragma omp parallel for shared(c, n, parallelize, chunk_size, x, i) num_threads(processors_count) private(r) \
        default(none) schedule(static) reduction(+ : s)
                for (r = i; r < n; r++) {
                    s += c[i][r] * x[r];
                }

                x[i] = (c[i][n] - s) / c[i][i];
            }
        }
    }
    return x;
}

std::vector<double> gauss(matrix c, bool parallelize, int processors_count, int chunk_size) {

    c = first_stage(c, parallelize, processors_count, chunk_size);
    std::vector<double> x = second_stage(c, parallelize, processors_count, chunk_size);
    return x;

}


int main(int argc, char *argv[]) {
    std::ofstream data_logger_file;
    data_logger_file.open("datalogger.log", std::ios_base::app); // append instead of overwrite
    std::string prompt;
    prompt = "N";
    int chunk_size, process_count;
    while (prompt == "N") {
        printf("Provide input file: ");
        std::cin >> prompt;
        fs::path input_path(fs::absolute(prompt));
        if (!fs::exists(input_path)) {
            printf("Provided file: %s doesnt exists, terminating.\n", input_path.string().c_str());
            exit(1);
        }
        datalogger::entry entry;
        entry.input_file = prompt;
        chunk_size = 0;
        printf("Specify number of processors:\n");
        std::cin >> process_count;
        printf("Specify chunk size for static scheduling (0 = default):\n");
        std::cin >> chunk_size;
        std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::string started_exec(30, '\0');
        std::strftime(&started_exec[0], started_exec.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        printf("Runtime started at: %s\n", started_exec.c_str());
        printf("Loading input csv file\n");
        matrix c = load_csv(input_path);
        printf("Calculating\n");
        double start_time = omp_get_wtime();
        std::vector<double> x_parallelized = gauss(c, true, process_count, chunk_size);
        double end_time = omp_get_wtime();
        double Tp = end_time - start_time;
        printf("Parallized it took: %f seconds\n", Tp);
        start_time = omp_get_wtime();
        std::vector<double> x_sequential = gauss(c, false, process_count, chunk_size);
        end_time = omp_get_wtime();
        double Ts = end_time - start_time;
        printf("Sequential it took: %f seconds\n", Ts);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(OUTPUT_TIME_DECIMAL_PRECISION);
        entry.Ts = Ts;
        entry.Tp = Tp;
        entry.chunk_size = chunk_size;
        entry.equation_count = c.size();
        entry.start_execution = started_exec;
        oss << "X_" << Ts << "_" << Tp << ".csv";
        printf("Saving output\n");
        save_matrix(x_parallelized, oss.str());
        data_logger_file << datalogger::to_string(entry) << std::endl;
        printf("Exit? [Y]/N:\n");

        std::cin >> prompt;
        if (prompt == "n") {
            prompt = "N";
        }
    }
    data_logger_file.close();
    return 0;
}