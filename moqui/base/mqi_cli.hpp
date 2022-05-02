#ifndef MQI_CLI_HPP
#define MQI_CLI_HPP

#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <valarray>
#include <vector>

#include <sys/stat.h>
//#include <errno.h>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace mqi
{

/**
 *  A command line interface for Unit tests
 *  cli is a top virtual class
 *  1. reading RT-Ion plan file and its type (plan or log).
 *  2. creating a geometries, patient, dosegrid, and beamline components of a machine
 *  3. creating beam sources of a machine
 */
class cli
{

protected:
    std::map<const std::string, std::vector<std::string>> parameters;

public:
    cli() {
        parameters = std::map<const std::string, std::vector<std::string>>({
          //option name,  {parameters}
          { "--dicom_path", {} },   //RTIP or RTIBTR
          { "--bname", {} },        //beam name
          { "--bnumber", {} },      //beam number
          { "--spots", {} },        //spot-id, obsolute, will be replaced by beamlets
          { "--beamlets", {} },     //beamlet-id
          { "--pph", {} },          //particles_per_history
          { "--sid", {} },          //beam generation distance from isocenter.
          { "--output_prefix",
            {} },                 //output file name, for empty, such as "", it printout to console
          { "--nhistory", {} },   // number of histories
          { "--pxyz", {} },       //position xyz w.r.t mother coordinate system
          { "--source_energy", {} },     //position xyz w.r.t mother coordinate system
          { "--energy_variance", {} },   //position xyz w.r.t mother coordinate system
          { "--rxyz", {} },              //position xyz w.r.t mother coordinate system
          { "--lxyz", {} },              //length of the volume for water phantom
          { "--pxyz", {} },              // position of the volume for water phantom
          { "--nxyz", {} },              // number of voxels in the water phantom
          { "--spot_position", {} },     // position of the beam
          { "--spot_size", {} },         // size of spot
          { "--spot_angles", {} },       //
          { "--spot_energy", {} },       // energy of the beam
          { "--histories", {} },         // the number of histories
          { "--threads", {} },           // the number of threads and blocks
          { "--score_variance", {} },    // ON/OFF for scoring variance
          { "--gpu_id", {} },            // ON/OFF for scoring variance
          { "--output_format", {} },     // raw, mha, or mhd for output format
          { "--random_seed", {} },       // raw, mha, or mhd for output format
          { "--phantom_path", {} }       // raw, mha, or mhd for output format
        });
    }
    ~cli() {
        ;
    }

    void
    read(int argc, char** argv) {

        std::cout << "# of arguments: " << argc << std::endl;
        // if (argc ==1) {print_help(argv[0]); exit(1);};

        for (int i = 1; i < argc; ++i) {
            //Find a parameter
            auto it = parameters.find(argv[i]);
            if (it != parameters.end()) {
                //Accumulate argv until it meets next option
                int j = i + 1;
                do {
                    if (std::string(argv[j]).compare(0, 2, "--") == 0) break;
                    it->second.push_back(argv[j]);
                    j++;
                } while (j < argc);
                //Print out options
                std::cout << it->first << " : ";
                for (auto parm : it->second)
                    std::cout << parm << " ";
                std::cout << std::endl;
            }
        }
    }
    virtual void
    print_help(char* s) {
        std::cout << "Usage:   " << s << " [-option] [argument]" << std::endl;
        std::cout << "options:  "
                  << "--dicom_path DICOM path " << std::endl;
        std::cout << "          "
                  << "--bname number" << std::endl;   //default 0
        std::cout << "          "
                  << "--bnumber beamname" << std::endl;   //no default
        std::cout << "          "
                  << "--spots i for single, i j for range, -1 for all."
                  << std::endl;   //e.g) i:i-th spot, i-j:i-th to j-th, -1: all
        std::cout << "          "
                  << "--beamlets i for single, i j for range, no options for all."
                  << std::endl;   //e.g) i:i-th spot, i-j:i-th to j-th, -1: all
        std::cout << "          "
                  << "--pph scale #-1 is 1 history/spot" << std::endl;   //default -1 -> 1 histories
        std::cout << "          "
                  << "--sid d" << std::endl;   //distance between beam and isocenter in z direction
        std::cout << "          "
                  << "--output_prefix prefix of output files " << std::endl;   //output file name
        std::cout << "          "
                  << "--nhistory number of histories"
                  << std::endl;   //position of phase-space coordinates system
        std::cout << "          "
                  << "--pxyz x y z" << std::endl;   //position of phase-space coordinates system
        std::cout << "          "
                  << "--source_energy e energy of source"
                  << std::endl;   //position of phase-space coordinates system
        std::cout << "          "
                  << "--energy_variance v variance of the source"
                  << std::endl;   //position of phase-space coordinates system
        std::cout << "          "
                  << "--rwxyz w x y z collimator angle, gantry angle, couch angle, iec2dicom angle "
                     "in degree"
                  << std::endl;   //position of phase-space coordinates systemu
        std::cout << "          "
                  << "--pxyz x y z \n";
        std::cout << "          "
                  << "--lxyz x y z \n";
        std::cout << "          "
                  << "--nxyz x y z \n";
        std::cout << "          "
                  << "--spot_size mean sigma \n";
        std::cout << "          "
                  << "--spot_position x y z \n";
        std::cout << "          "
                  << "--spot_angles colimater gantry couch iec2dicom \n";
        std::cout << "          "
                  << "--spot_energy mean sigma \n";
        std::cout << "          "
                  << "--histories n \n";
        std::cout << "          "
                  << "--threads blocks threads_per_block \n";
        std::cout << "          "
                  << "--score_variance 0 for not scoring 1 for scoring \n";
    }
    const std::vector<std::string>
    operator[](const std::string& t) {
        return parameters[t];
    }
};
}   // namespace mqi

#endif
