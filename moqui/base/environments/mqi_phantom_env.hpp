#ifndef MQI_PHANTOM_ENV_HPP
#define MQI_PHANTOM_ENV_HPP

#include <cstring>
#include <iomanip>
#include <math.h>
#include <moqui/base/environments/mqi_xenvironment.hpp>
#include <moqui/base/materials/mqi_patient_materials.hpp>
#include <moqui/base/mqi_distributions.hpp>
#include <moqui/base/mqi_io.hpp>
#include <moqui/base/mqi_threads.hpp>
#include <moqui/base/scorers/mqi_scorer_energy_deposit.hpp>

namespace mqi
{

template<typename R>
class phantom_env : public x_environment<R>
{
public:
    mqi::vec3<R>   lxyz;   ///< size of water phantom box
    mqi::vec3<int> nxyz;   ///< number of voxels
    mqi::vec3<R>   pos;    ///< water phantom position (center)

    std::array<R, 3> spot_position;
    std::array<R, 2> spot_size;
    std::array<R, 2> spot_energy;
    std::array<R, 4> spot_angles;
    size_t           n_histories = 0;   ///0.1M
    bool             score_variance;
    ///< material_list 0 -> air, 1->water
    std::array<uint32_t, 2> threads;
    int                     random_seed  = 0;
    std::string             phantom_path = "";

public:
    CUDA_HOST
    phantom_env(mqi::cli& cli) : x_environment<R>() {
        if (cli["--gpu_id"].size() >= 1) {
            this->gpu_id = std::stoi(cli["--gpu_id"][0]);
        } else {
            this->gpu_id = 0;
            printf("gpu_id 0\n");
        }
#if defined(__CUDACC__)
        cudaSetDevice(this->gpu_id);
#endif
        ///< Geometry, nxyz, lxyz, pxyz
        ///< - dimension
        if (cli["--lxyz"].size() >= 1) {
            lxyz.x = std::stof(cli["--lxyz"][0]);
            lxyz.y = std::stof(cli["--lxyz"][1]);
            lxyz.z = std::stof(cli["--lxyz"][2]);
        } else {
            lxyz.x = 512.0;
            lxyz.y = 512.0;
            lxyz.z = 400.0;
        }
        ///< - center position of water phantom
        if (cli["--pxyz"].size() >= 1) {
            pos.x = std::stof(cli["--pxyz"][0]);
            pos.y = std::stof(cli["--pxyz"][1]);
            pos.z = std::stof(cli["--pxyz"][2]);
        } else {
            pos.x = -256.0;
            //            pos.x = -0.5;
            pos.y = 0.0;
            pos.z = 0.0;
        }
        ///< - number of voxels
        if (cli["--nxyz"].size() >= 1) {
            nxyz.x = std::stoi(cli["--nxyz"][0]);
            nxyz.y = std::stoi(cli["--nxyz"][1]);
            nxyz.z = std::stoi(cli["--nxyz"][2]);
        } else {
            nxyz.x = 512;   /// 1 mm
            nxyz.y = 512;   /// 1 mm
            nxyz.z = 200;   ///2 mm
        }

        ///< Beam position
        if (cli["--spot_position"].size() >= 1) {
            spot_position[0] = std::stof(cli["--spot_position"][0]);
            spot_position[1] = std::stof(cli["--spot_position"][1]);
            spot_position[2] = std::stof(cli["--spot_position"][2]);
        } else {
            spot_position[0] = 1.0;
            spot_position[1] = 0.0;
            spot_position[2] = 0.0;
        }

        ///< Beam position
        if (cli["--spot_size"].size() >= 1) {
            spot_size[0] = std::stof(cli["--spot_size"][0]);
            spot_size[1] = std::stof(cli["--spot_size"][1]);

        } else {
            spot_size[0] = 0.0;
            spot_size[1] = 0.0;
        }

        ///< beam energy
        if (cli["--spot_energy"].size() >= 1) {
            const auto& t  = cli["--spot_energy"];
            spot_energy[0] = static_cast<R>(std::stof(t[0]));
            spot_energy[1] = static_cast<R>(std::stof(t[1]));
        } else {
            spot_energy[0] = 230;
            spot_energy[1] = 0.0;
        }

        if (cli["--histories"].size() >= 1) {
            n_histories = std::stoi(cli["--histories"][0]);
        } else {
            n_histories = 10000;
        }

        if (cli["--spot_angles"].size() >= 1) {
            const auto& t  = cli["--spot_angles"];
            spot_angles[0] = static_cast<R>(std::stof(t[0]));   //colli
            spot_angles[1] = static_cast<R>(std::stof(t[1]));   //gantry
            spot_angles[2] = static_cast<R>(std::stof(t[2]));   //couch
            spot_angles[3] = static_cast<R>(std::stof(t[3]));   //iec2dicom
        } else {
            spot_angles[0] = 0.0;
            spot_angles[1] = 0.0;
            spot_angles[2] = 0.0;
            spot_angles[3] = 0.0;
        }
        int particles_per_thread = 0;

        switch (cli["--threads"].size()) {
        case 0:
            threads[0]           = thread_limit;
            threads[1]           = (int) std::ceil(n_histories * 1.0 / threads[0]);
            particles_per_thread = (int) std::ceil(n_histories * 1.0 / (threads[0] * threads[1]));
            if (n_histories % (threads[0] * threads[1] * particles_per_thread) > 0) {
                threads[1] += 1;   // increase block size if there is any remainder
            }
            if (threads[1] > block_limit) { threads[1] = block_limit; }
            break;
        case 1:
            threads[0] = std::stoi(cli["--threads"][0]);
            threads[1] = 1;
            break;
        case 2:
            threads[0] = std::stoi(cli["--threads"][0]);
            threads[1] = std::stoi(cli["--threads"][1]);
            break;
        }

        if (cli["--score_variance"].size() >= 1) {
            std::string v        = cli["--score_variance"][0];
            this->score_variance = strcasecmp(v.c_str(), "true") == 0 || std::atoi(v.c_str()) != 0;
        } else {
            this->score_variance = false;
        }

        if (cli["--output_prefix"].size() >= 1) {
            this->output_path = cli["--output_prefix"][0];
            printf("%s\n", this->output_path.c_str());
        } else {
            throw std::runtime_error("output_prefix is required.");
        }

        if (cli["--phantom_path"].size() >= 1) {
            this->phantom_path = cli["--phantom_path"][0];
            printf("phantom path: %s\n", this->phantom_path.c_str());
        } else {
            throw std::runtime_error("phantom_path is required.");
        }

        if (cli["--random_seed"].size() >= 1) {
            this->random_seed = std::stoi(cli["--random_seed"][0]);
            printf("random seed input %d\n", this->random_seed);
        } else {
            this->random_seed =
              static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count());
        }
        printf("random seed %d\n", this->random_seed);
        this->beam_rng.seed(this->random_seed);
    }

    CUDA_HOST
    ~phantom_env() {
        ;
    }

    CUDA_HOST
    virtual void
    print_parameters() {
        printf("Position of water phantom: %f %f %f\n", pos.x, pos.y, pos.z);
        printf("Size of water phantom: %f %f %f\n", lxyz.x, lxyz.y, lxyz.z);
        printf("Number of voxels: %d %d %d\n", nxyz.x, nxyz.y, nxyz.z);
        printf("Spot position: %f %f %f\n", spot_position[0], spot_position[1], spot_position[2]);
        printf("Spot size: %f %f\n", spot_size[0], spot_size[1]);
        printf("Spot angles %f %f %f\n", spot_angles[0], spot_angles[1], spot_angles[2]);
        printf("Spot energies %f %f\n", spot_energy[0], spot_energy[1]);
        printf("Histories %lu\n", n_histories);
        printf("Number of threads %d %d\n", threads[0], threads[1]);
        printf("Score variance %d\n", score_variance);
    }

    CUDA_HOST
    virtual void
    setup_materials() {
        this->n_materials  = 2;
        this->materials    = new mqi::material_t<R>[this->n_materials];
        this->materials[0] = mqi::air_t<R>();
        this->materials[1] = mqi::h2o_t<R>();
        ///*
        for (uint16_t i = 0; i < 2; ++i) {
            printf(
              "%d: %f IeV, %f g/cm^3\n", i, this->materials[i].Iev, this->materials[i].rho_mass);
        }
        //*/
    }

    CUDA_HOST
    virtual void
    setup_beamsource() {

        mqi::coordinate_transform<R> p_coord(spot_angles, { 0, 0, 0 });   //angles, isocenter
        /// Source direction is -Z
        mqi::vec3<R> dir(0, 0, -1);
        ///< Beamlet phse-space distribution
        std::array<R, 6> beamlet_mean = {
            spot_position[0], spot_position[1], spot_position[2], dir.x, dir.y, dir.z
        };
        std::array<R, 6> beamlet_sigm = { spot_size[0], spot_size[1], 0.0, 0.0, 0.0, 0.0 };
        std::array<R, 2> corr         = { 0.0, 0.0 };
        auto             phsp = new mqi::phsp_6d_uniform<R>(beamlet_mean, beamlet_sigm, corr);

        ///< Beamlet energy distribution
        auto energy = new mqi::const_1d<R>({ spot_energy[0] }, { spot_energy[1] });
        this->beamsource.append_beamlet(mqi::beamlet<R>(energy, phsp), n_histories, p_coord);

        ///< generate initial vertices
        printf("total histories %lu\n", this->beamsource.total_histories());
        uint32_t h0    = 0;
        uint32_t h1    = this->beamsource.total_histories();
        this->vertices = new mqi::vertex_t<R>[h1 - h0];

        for (size_t i = 0; i < (h1 - h0); ++i) {
            auto bl           = this->beamsource(h0 + i);
            this->vertices[i] = bl(&this->beam_rng);
        }
    }

    CUDA_HOST
    virtual void
    setup_world() {
        this->world = new mqi::node_t<R>;

        ///< By default, let's set +- 30 cm as world volume
        const R hl   = 600.0;
        R       x[2] = { static_cast<R>(-0.5 * hl), static_cast<R>(0.5 * hl) };
        R       y[2] = { static_cast<R>(-0.5 * hl), static_cast<R>(0.5 * hl) };
        R       z[2] = { static_cast<R>(-0.5 * hl), static_cast<R>(0.5 * hl) };

        this->world->geo = new mqi::grid3d<mqi::density_t, R>(x, 2, y, 2, z, 2);

        this->world->geo->fill_data(0.00000121);   //air
        this->world->n_scorers  = 0;
        this->world->scorers    = nullptr;
        this->world->n_children = 1;
        this->world->children   = new node_t<R>*[this->world->n_children];

        ///< create a child
        node_t<R>* phantom       = new node_t<R>;
        this->world->children[0] = phantom;
        x[0]                     = pos.x - 0.5 * lxyz.x;
        x[1]                     = pos.x + 0.5 * lxyz.x;
        y[0]                     = pos.y - 0.5 * lxyz.y;
        y[1]                     = pos.y + 0.5 * lxyz.y;
        z[0]                     = pos.z - 0.5 * lxyz.z;
        z[1]                     = pos.z + 0.5 * lxyz.z;

        phantom->geo = new grid3d<mqi::density_t, R>(
          x[0], x[1], nxyz.x + 1, y[0], y[1], nxyz.y + 1, z[0], z[1], nxyz.z + 1);

        density_t*                 rho_mass     = new density_t[nxyz.x * nxyz.y * nxyz.z];
        mqi::density_t             bone_density = 1.0;
        mqi::patient_material_t<R> patient_material;
        int16_t*                   ph = new int16_t[nxyz.x * nxyz.y * nxyz.z];
        std::ifstream              ph_fid(this->phantom_path, std::ios::in | std::ios::binary);
        ph_fid.read((char*) (&ph[0]), nxyz.x * nxyz.y * nxyz.z * sizeof(ph[0]));
        ph_fid.close();
        for (int i = 0; i < nxyz.x * nxyz.y * nxyz.z; i++) {
            rho_mass[i] = patient_material.hu_to_density(ph[i]);
        }
        phantom->geo->set_data(rho_mass);   //// Material conversion function required

        ///< TODO : compile error (type of dE)

        phantom->n_scorers = 1;   //4;
        phantom->scorers   = new scorer<R>*[phantom->n_scorers];
        fp_compute_hit<R> fp0;
#if defined(__CUDACC__)
        cudaMemcpyFromSymbol(&fp0, mqi::Dw_pointer, sizeof(fp_compute_hit<R>));
#else
        fp0             = mqi::dose_to_water;
#endif
        phantom->scorers[0] = new mqi::scorer<R>("water_dE_total", nxyz.x * nxyz.y * nxyz.z, fp0);
        mqi::key_value* deposit0 = new mqi::key_value[phantom->scorers[0]->max_capacity_];
        std::memset(deposit0, 0xff, sizeof(mqi::key_value) * phantom->scorers[0]->max_capacity_);
        init_table(deposit0, phantom->scorers[0]->max_capacity_);

        phantom->scorers[0]->data_           = deposit0;
        phantom->scorers[0]->score_variance_ = this->score_variance;
        phantom->scorers[0]->roi_ = new mqi::roi_t(mqi::DIRECT, nxyz.x * nxyz.y * nxyz.z);

        if (this->score_variance) {
            mqi::key_value* count          = new mqi::key_value[phantom->scorers[0]->max_capacity_];
            mqi::key_value* vox_mean       = new mqi::key_value[phantom->scorers[0]->max_capacity_];
            mqi::key_value* vox_variance   = new mqi::key_value[phantom->scorers[0]->max_capacity_];
            phantom->scorers[0]->count_    = count;
            phantom->scorers[0]->mean_     = vox_mean;
            phantom->scorers[0]->variance_ = vox_variance;
        }
        mc::mc_score_variance = this->score_variance;
    }

    CUDA_HOST
    virtual void
    setup_scorers() {}

    CUDA_HOST
    virtual void
    run() {
        auto     start  = std::chrono::high_resolution_clock::now();
        uint32_t h0     = 0;
        uint32_t h1     = this->beamsource.total_histories();
        this->num_spots = this->beamsource.total_beamlets();
        mqi::thrd_t* worker_threads;
        uint32_t*    tracked_particles              = new uint32_t[1];
        tracked_particles[0]                        = 0;
        key_t*                 scorer_offset_vector = new key_t[h1 - h0];
        uint32_t*              h_count              = new uint32_t[nxyz.x];
        int                    histories_per_spot = 0, idx = 0;
        unsigned long long int scorer_offset = nxyz.x * nxyz.y * nxyz.z;
        printf("num spots %d\n", this->num_spots);
        for (int spot_id = 0; spot_id < this->num_spots; spot_id++) {
            auto bl            = this->beamsource[spot_id];
            histories_per_spot = std::get<1>(bl);
            for (int vtx_id = 0; vtx_id < histories_per_spot; vtx_id++) {
                scorer_offset_vector[idx] = spot_id;
                idx++;
            }
        }
#if defined(__CUDACC__)
        size_t free = 0, total = 0;
        mc::upload_vertices(this->vertices, mc::mc_vertices, h0, h1);
        cudaMemGetInfo(&free, &total);
        printf("Vertices total memory %lu MB Free memory %lu MB\n",
               total / (1024 * 1024),
               free / (1024 * 1024));
        cudaError_t err = cudaGetLastError();   // add
        if (err != cudaSuccess) {
            std::cout << "CUDA error (upload_vertices): " << cudaGetErrorString(err) << std::endl;
            exit(-1);
        }
        key_t* d_scorer_offset_vector;
        free  = 0;
        total = 0;
        mc::upload_scorer_offset_vector(scorer_offset_vector, d_scorer_offset_vector, h1 - h0);
        cudaMemGetInfo(&free, &total);
        printf("Scorer offset total memory %lu MB Free memory %lu MB\n",
               total / (1024 * 1024),
               free / (1024 * 1024));
        cudaDeviceSynchronize();
        err = cudaGetLastError();   // add
        if (err != cudaSuccess) {
            std::cout << "CUDA error (upload scoerer offset vector): " << cudaGetErrorString(err)
                      << std::endl;
            exit(-1);
        }
        gpu_err_chk(cudaMalloc(&worker_threads, threads[1] * threads[0] * sizeof(mqi::thrd_t)));

        free  = 0;
        total = 0;
        initialize_threads<<<threads[1], threads[0]>>>(worker_threads, threads[1] * threads[0], 0);
        cudaMemGetInfo(&free, &total);
        printf("After thread init total memory %lu MB Free memory %lu MB\n",
               total / (1024 * 1024),
               free / (1024 * 1024));
        //        initialize_threads<<<threads[1], threads[0]>>>(worker_threads, 0, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error (initialize threads): " << cudaGetErrorString(err)
                      << std::endl;
            exit(-1);
        }
        uint32_t* d_tracked_particles;
        gpu_err_chk(cudaMalloc(&d_tracked_particles, sizeof(tracked_particles)));
        gpu_err_chk(cudaMemcpy(d_tracked_particles,
                               tracked_particles,
                               sizeof(tracked_particles),
                               cudaMemcpyHostToDevice));

        printf("blocks %d threads %d\n", threads[1], threads[0]);
        free  = 0;
        total = 0;
        mc::transport_particles_patient<R><<<threads[1], threads[0]>>>(
          worker_threads, mc::mc_world, mc::mc_vertices, h1, d_tracked_particles);
        cudaMemGetInfo(&free, &total);
        printf("After run total memory %lu MB Free memory %lu MB\n",
               total / (1024 * 1024),
               free / (1024 * 1024));
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error (transport particle table): " << cudaGetErrorString(err)
                      << std::endl;
            exit(-1);
        }
        gpu_err_chk(cudaMemcpy(tracked_particles,
                               d_tracked_particles,
                               sizeof(tracked_particles),
                               cudaMemcpyDeviceToHost));

        gpu_err_chk(cudaFree(d_scorer_offset_vector));
        gpu_err_chk(cudaFree(worker_threads));
        gpu_err_chk(cudaFree(d_tracked_particles));
        gpu_err_chk(cudaFree(mc::mc_vertices));

//            std::cout << "Number of particles tracked " << tracked_particles[0] << std::endl;
#else
        mc::mc_world    = this->world;
        mc::mc_vertices = this->vertices;
        worker_threads  = new mqi::thrd_t[1];
        mc::transport_particles_table<R>(
          worker_threads, mc::mc_world, mc::mc_vertices, h1, tracked_particles);
#endif
        std::cout << "Number of particles tracked " << tracked_particles[0] << std::endl;
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        printf("Run done %f ms\n", duration.count());
    }

    CUDA_HOST
    void
    print_reshaped_results() {
        auto start = std::chrono::high_resolution_clock::now();

        uint32_t         vol_size;
        mqi::vec3<ijk_t> dim;
        R*               reshaped_data;
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                dim           = this->world->children[c_ind]->geo->get_nxyz();
                vol_size      = dim.x * dim.y * dim.z;
                reshaped_data = this->reshape_data(c_ind, s_ind, dim);
                for (int v_ind = 0; v_ind < vol_size; v_ind++) {
                    printf("%d %f\n", v_ind, reshaped_data[v_ind]);
                }
                delete[] reshaped_data;
            }
        }
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        printf("Reshape and save done %f ms\n", duration.count());
    }
};

}   // namespace mqi

#endif
