#include <cassert>
#include <chrono>
#include <iostream>

#include <moqui/base/environments/mqi_tps_env.hpp>
#include <moqui/base/mqi_cli.hpp>

typedef float phsp_t;

int
main(int argc, char* argv[]) {
    ///< construct a treatment planning system environment
    auto        start = std::chrono::high_resolution_clock::now();
    std::string input_file;

    if (argc > 1) {
        input_file = argv[1];
    } else {
        input_file = "./moqui_tps.in";
    }
    mqi::tps_env<phsp_t> myenv(input_file);
    myenv.initialize_and_run();

    auto                                      stop     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Time taken by MC engine: " << duration.count() << " milli-seconds\n";
    return 0;
}
