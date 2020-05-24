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
        std::string start_day;
        std::string start_time;
        std::string input_file;
        int equation_count{};
        double Ts{};
        double Tp{};
        int process_count{};
        int chunk_size{};
    };

    std::ostream &operator<<(std::ostream &os, entry const &arg) {
        os << std::setprecision(OUTPUT_TIME_DECIMAL_PRECISION);
        os << arg.start_day.c_str() << ";" << arg.start_time.c_str() << ";" << arg.input_file.c_str() << ";"
           << arg.equation_count << ";" << arg.Ts << ";" << arg.Tp << arg.process_count << ";" << arg.chunk_size;
        return os;
    }

    std::string to_string(entry const &arg) {
        std::ostringstream ss;
        ss << arg;
        return std::move(ss).str();
    }

    std::string header() {
        return std::string("DATE;TIME;INPUT_FILE;EQUATION_COUNT;Ts;Tp;PROCESS_COUNT;CHUNK_SIZE");
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
#pragma omp parallel shared(c, n, chunk_size, r, processors_count) num_threads(processors_count) private(i, j) default(none)
            printf("First stage using static schedule processor number: %d chunks: %d\n",
                   omp_get_thread_num(), chunk_size);
            for (r = 0; r < n - 1; r++) {
#pragma omp for ordered schedule(static, chunk_size)
                for (i = r + 1; i < n; i++) {
                    for (j = r + 1; j < n + 1; j++) {
                        c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
                    }
                }
            }
        } else {
#pragma omp parallel shared(c, n, chunk_size, r, processors_count) num_threads(processors_count) private(i, j) default(none)
            {
                printf("First stage using static schedule processor number: %d\n", omp_get_thread_num());
                for (r = 0; r < n - 1; r++) {
#pragma omp for ordered schedule(static)
                    for (i = r + 1; i < n; i++) {
                        for (j = r + 1; j < n + 1; j++) {
                            c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
                        }
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
    bool datalogger_already_created = fs::exists("datalogger.log");
    std::ofstream data_logger_file;
    data_logger_file.open("datalogger.log", std::ios_base::app);
    if(!datalogger_already_created) {
        data_logger_file << datalogger::header() << std::endl;
    }
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
        std::string start_day(30, '\0');
        std::string start_time(30, '\0');
        std::strftime(&start_day[0], start_day.size(), "%Y-%m-%d", std::localtime(&now));
        std::strftime(&start_time[0], start_time.size(), "%H:%M:%S", std::localtime(&now));
        printf("Runtime started at: %s %s\n", start_day.c_str(), start_time.c_str());
        printf("Loading input csv file\n");
        matrix c = load_csv(input_path);
        printf("Calculating\n");
        double execution_start_time = omp_get_wtime();
        std::vector<double> x_parallelized = gauss(c, true, process_count, chunk_size);
        double execution_end_time = omp_get_wtime();
        double Tp = execution_end_time - execution_start_time;
        printf("Tp: %f seconds\n", Tp);
        execution_start_time = omp_get_wtime();
        std::vector<double> x_sequential = gauss(c, false, process_count, chunk_size);
        execution_end_time = omp_get_wtime();
        double Ts = execution_end_time - execution_start_time;
        printf("Ts: %f seconds\n", Ts);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(OUTPUT_TIME_DECIMAL_PRECISION);
        entry.Ts = Ts;
        entry.Tp = Tp;
        entry.chunk_size = chunk_size;
        entry.equation_count = c.size();
        entry.start_day = start_day;
        entry.start_time = start_time;
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