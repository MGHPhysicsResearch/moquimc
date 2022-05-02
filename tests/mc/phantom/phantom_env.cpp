#include <iostream>

#include <moqui/base/environments/mqi_phantom_env.hpp>

int
main(int argc, char* argv[]) {
    auto     start = std::chrono::high_resolution_clock::now();
    mqi::cli cl_opts;
    cl_opts.read(argc, argv);

    mqi::phantom_env<mqi::phsp_t> myenv(cl_opts);
    myenv.initialize();

    myenv.run();
    myenv.finalize();
    myenv.save_reshaped_files();

    auto                                      stop     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Time taken by MC engine: " << duration.count() << " milli-seconds\n";
    return 0;
}