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
const int MIN_SECOND_STAGE_LOOPS = 500;

using matrix = std::vector<std::vector<double>>;
namespace fs = std::filesystem;


void log_and_stdout(std::ostringstream &oss, std::ofstream &data_logger_file) {
    std::cout << oss.str();
    data_logger_file << oss.str();
    oss.str(std::string());
}

std::string set_schedule_type(std::string &chosen_type, int chunk_size) {
    if (chosen_type == "static") {
        omp_set_schedule(omp_sched_static, chunk_size);
    } else if (chosen_type == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, chunk_size);
    } else if (chosen_type == "guided") {
        omp_set_schedule(omp_sched_guided, chunk_size);
    } else {
        chosen_type = "auto";
        omp_set_schedule(omp_sched_auto, chunk_size);
    }
    std::ostringstream oss;
    oss << chosen_type;
    if (chunk_size > 0 && chosen_type != "auto") {
        oss << " with " << chunk_size << " chunks";
    }
    return oss.str();
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

matrix first_stage_sequentially(matrix &c, std::ofstream &data_logger_file) {
    int n = c.size();
    int r, i, j;
    std::cout << "First stage sequentially\n";
    for (r = 0; r < n - 1; r++) {
        for (i = r + 1; i < n; i++) {
            for (j = r + 1; j < n + 1; j++) {
                c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
            }
        }
    }
    return c;
}

std::vector<double> second_stage_sequentially(const matrix &c, std::ofstream &data_logger_file) {
    int n = c.size();
    std::vector<double> x(n);
    double s = 0.0;
    int i, r;
    x[n - 1] = c[n - 1][n] / c[n - 1][n - 1];
    std::cout << "Second stage sequentially\n";
    for (i = n - 2; i >= 0; i--) {
        s = 0;
        for (r = i; r < n; r++) {
            s = s + c[i][r] * x[r];
        }
        x[i] = (c[i][n] - s) / c[i][i];
    }
    return x;
}

matrix &first_stage(matrix &c, int processors_count, const std::string &schedule_type_info, std::ofstream &data_logger_file) {
    int n = c.size();
    int r, i, j;
    bool first_run = true;
    std::vector<bool> first_runs(processors_count, true);
    std::ostringstream oss;
    oss << "First stage using schedule: " << schedule_type_info.c_str() << std::endl;
    log_and_stdout(oss, data_logger_file);
    for (r = 0; r < n - 1; r++) {
#pragma omp parallel shared(c, n, r, data_logger_file) num_threads(processors_count) firstprivate(first_run) private(i, j, oss) default(none)
#pragma omp for schedule(runtime)
        for (i = r + 1; i < n; i++) {
            if ((r == 0) & first_run) {
#pragma omp critical
                {
                    oss  << "Processor " << omp_get_thread_num() << " started first stage with i: " << i << std::endl;
                    printf(oss.str().c_str());
                    data_logger_file << oss.str();
                }
                first_run = false;
            }
            for (j = r + 1; j < n + 1; j++) {
                c[i][j] = c[i][j] - (c[i][r] / c[r][r] * c[r][j]);
            }
        }
        first_run = false;
    }
    return c;
}


std::vector<double>
second_stage(const matrix &c, int processors_count, const std::string &schedule_type_info, std::ofstream &data_logger_file) {
    int n = c.size();
    std::vector<double> x(n);
    double s;
    int i, r;
    bool first_run = true;

    x[n - 1] = c[n - 1][n] / c[n - 1][n - 1];
    std::ostringstream oss;
    oss << "Second stage in parallel using schedule: " << schedule_type_info.c_str() << std::endl;
    log_and_stdout(oss, data_logger_file);
    std::vector<bool> first_runs(processors_count, true);

    for (i = n - 2; i >= 0; i--) {
        s = 0.0;
/* if(i + MIN_SECOND_STAGE_LOOPS < n) wynika z tego że zrównoleglenie dla dużej ilości pierwszych pętli nie ma sensu,
 * więcej czasu jest zużywane na utworzenie procesu i jego zarządzanie niż jest to warte dla zrónoleglenia
 * dla zaledwie kilka(naście) przebiegów. Więc zrównoleglenie w tym miejscu uruchomi się tylko po pewnej ilości przebiegów,
 * lub wcale - dla małej ilości danych wejściowych możliwa jest kalibracja momentu rozpoczęcia zrównoleglenia poprzez
 * zmodyfikowanie stałej MIN_SECOND_STAGE_LOOPS
*/
#pragma omp parallel for schedule(runtime) reduction(+ : s) shared(c, n, x, i, first_runs, data_logger_file) \
num_threads(processors_count) firstprivate(first_run) \
private(r, oss) default(none) if (i + MIN_SECOND_STAGE_LOOPS < n)
        for (r = i; r < n; r++) {
            if ((i == 0) & first_run) {
#pragma omp critical
                {
                    oss << "Processor " << omp_get_thread_num() << " ended second stage " << std::endl;
                    printf(oss.str().c_str());
                    data_logger_file << oss.str();
                    first_run = false;
                }
            }
            s += c[i][r] * x[r];
        }
        x[i] = (c[i][n] - s) / c[i][i];
    }
    return x;
}

std::vector<double>
gauss(matrix c, bool parallelize, int processors_count, const std::string &schedule_type_info,
      std::ofstream &data_logger_file) {
    std::vector<double> x;
    if (!parallelize) {
        c = first_stage_sequentially(c, data_logger_file);
        x = second_stage_sequentially(c, data_logger_file);
    } else {
        c = first_stage(c, processors_count, schedule_type_info, data_logger_file);
        x = second_stage(c, processors_count, schedule_type_info, data_logger_file);
    }
    return x;
}


int main(int argc, char *argv[]) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(OUTPUT_TIME_DECIMAL_PRECISION);
    std::ofstream data_logger_file;
    data_logger_file.open("datalogger.log", std::ios_base::app);
    std::string prompt, schedule_type, schedule_type_info;
    prompt = "N";
    int chunk_size, process_count;
    while (prompt == "N") {
        printf("Provide input file:\n");
        std::cin >> prompt;
        fs::path input_path(fs::absolute(prompt));
        if (!fs::exists(input_path)) {
            printf("Provided file: %s doesnt exists, terminating", input_path.string().c_str());
            exit(1);
        }
        oss << "Input file: " << prompt << std::endl;
        chunk_size = 0;
        printf("Specify number of processors:\n");
        std::cin >> process_count;
        printf("Specify schedule type (static, dynamic, guided, auto[default]):\n");
        std::cin >> schedule_type;
        if (schedule_type != "auto") {
            printf("Specify chunk size scheduling (0 = default):\n");
            std::cin >> chunk_size;
        }
        schedule_type_info = set_schedule_type(schedule_type, chunk_size);
        oss << "Selected schedule: " << schedule_type_info << std::endl;
        log_and_stdout(oss, data_logger_file);
        std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::string start_day(30, '\0');
        std::string start_time(30, '\0');
        std::strftime(&start_day[0], start_day.size(), "%Y-%m-%d", std::localtime(&now));
        std::strftime(&start_time[0], start_time.size(), "%H:%M:%S", std::localtime(&now));
        oss << "Runtime started at: " << start_day.c_str() << " " << start_time.c_str() << std::endl;
        log_and_stdout(oss, data_logger_file);
        printf("Loading input csv file\n");
        matrix c = load_csv(input_path);
        std::cout << "Calculating\n";
        data_logger_file.flush();
        double execution_start_time = omp_get_wtime();
        std::vector<double> x_parallelized = gauss(c, true, process_count, schedule_type_info, data_logger_file);
        double execution_end_time = omp_get_wtime();
        double Tp = execution_end_time - execution_start_time;
        oss.str(std::string());
        oss << "Tp: " << Tp << " seconds\n";
        log_and_stdout(oss, data_logger_file);
        data_logger_file.flush();
        execution_start_time = omp_get_wtime();
        std::vector<double> x_sequential = gauss(c, false, process_count, schedule_type_info, data_logger_file);
        execution_end_time = omp_get_wtime();
        double Ts = execution_end_time - execution_start_time;
        oss.str(std::string());
        oss << "Ts: " << Ts << " seconds\n";
        log_and_stdout(oss, data_logger_file);
        data_logger_file.flush();
        oss << "X_" << Ts << "_" << Tp << ".csv";
        save_matrix(x_parallelized, oss.str());
        oss.str(std::string());
        oss << "file has been saved under name: " << "X_" << Ts << "_" << Tp << ".csv" << std::endl;
        log_and_stdout(oss, data_logger_file);
        printf("Exit? [Y]/N:\n");

        std::cin >> prompt;
        if (prompt == "n") {
            prompt = "N";
        }
    }
    return 0;
}

