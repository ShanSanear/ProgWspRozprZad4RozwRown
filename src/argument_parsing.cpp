#include "../include/cxxopts.hpp"

void show_help_and_exit(const cxxopts::Options &options, int exit_code) {
    std::cout << options.help({""}) << std::endl;
    exit(exit_code);
}

auto parse_arguments(int argc, char **argv) {
    try {
        cxxopts::Options options(argv[0]);
        options
        .positional_help("[input]")
        .show_positional_help();
        options.add_options()
                ("input", "Path to input csv file", cxxopts::value<std::string>())
                ("positional", "Excessive positional arguments", cxxopts::value<std::vector<std::string>>())
                ("help", "Print help");
        options.parse_positional({"input", "positional"});
        auto result = options.parse(argc, argv);
        const auto& arguments = result.arguments();
        if (result.count("help") | (arguments.empty())) {
            show_help_and_exit(options, 0);
        }
        if (result.count("input") != 1) {
            printf("Input file has not been provided\n");
            show_help_and_exit(options, 1);
        }
        return result;
    }
    catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

}