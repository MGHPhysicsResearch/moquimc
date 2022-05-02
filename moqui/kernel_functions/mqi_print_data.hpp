#ifndef MQI_PRINT_DATA_HPP
#define MQI_PRINT_DATA_HPP

#include <moqui/base/mqi_material.hpp>
#include <moqui/base/mqi_node.hpp>
namespace mc
{
template<typename R>
CUDA_GLOBAL void
print_materials(mqi::material_t<R>* m, uint16_t n_m);
template<typename R>
CUDA_DEVICE void
print_node_specification_gpu(mqi::node_t<R>* node);
template<typename R>
CUDA_GLOBAL void
print_node_specification(mqi::node_t<R>* node);

template<typename R>
CUDA_GLOBAL void
print_materials(mqi::material_t<R>* m, uint16_t n_m) {
    for (uint16_t i = 0; i < n_m; ++i) {
        printf("%d: %f IeV, %f g/cm^3\n", i, m[i].Iev, m[i].rho_mass);
    }
}

template<typename R>
CUDA_DEVICE void
print_node_specification_gpu(mqi::node_t<R>* node) {
    mqi::vec3<mqi::ijk_t> dim   = node->geo->get_nxyz();
    const uint32_t        total = dim.x * dim.y * dim.z;

    /// Code for checking X/Y/Z position and edge data
    const R*       x  = node->geo->get_x_edges();
    const R*       y  = node->geo->get_y_edges();
    const R*       z  = node->geo->get_z_edges();
    const uint16_t nx = (dim.x < 5) ? dim.x : 5;
    const uint16_t ny = (dim.y < 5) ? dim.y : 5;
    const uint16_t nz = (dim.z < 5) ? dim.z : 5;

    printf("X: \n");
    for (uint16_t i = 0; i < nx; ++i) {
        printf("%.3f ", x[i]);
    }
    printf("... %.3f\n", x[dim.x]);

    printf("Y: \n");
    for (uint16_t i = 0; i < ny; ++i) {
        printf("%.3f ", y[i]);
    }
    printf("... %.3f\n", y[dim.y]);

    printf("Z: \n");
    for (uint16_t i = 0; i < nz; ++i) {
        printf("%.3f ", z[i]);
    }
    printf("... %.3f\n", z[dim.z]);

    mqi::vec3<mqi::ijk_t> idx(0, 0, 0);
    mqi::material_id      v = (*node->geo)[idx];
    //    mqi::material_t<R> v = (*node->geo)[idx];
    printf("From phantom: (%d, %d, %d): %f\n", idx.x, idx.y, idx.z, v);

    for (int i = 0; i < node->n_children; ++i) {
        printf("child node: %d, %p\n", i, node->children[i]);
        print_node_specification_gpu<R>(node->children[i]);
    }
}

template<typename R>
CUDA_GLOBAL void
print_node_specification(mqi::node_t<R>* node) {
    const mqi::vec3<mqi::ijk_t> dim   = node->geo->get_nxyz();
    const uint32_t              total = dim.x * dim.y * dim.z;

    /// Code for checking X/Y/Z position and edge data
    const R*       x  = node->geo->get_x_edges();
    const R*       y  = node->geo->get_y_edges();
    const R*       z  = node->geo->get_z_edges();
    const uint16_t nx = (dim.x < 5) ? dim.x : 5;
    const uint16_t ny = (dim.y < 5) ? dim.y : 5;
    const uint16_t nz = (dim.z < 5) ? dim.z : 5;

    printf("X: \n");
    for (uint16_t i = 0; i < nx; ++i) {
        printf("%.3f ", x[i]);
    }
    if (nx < dim.x) {
        printf("... ");
        for (uint16_t i = dim.x - nx; i < dim.x + 1; ++i) {
            printf("%.3f ", x[i]);
        }
        printf("\n");
    } else {
        printf("... %.3f\n", x[dim.x]);
    }

    printf("Y: \n");
    for (uint16_t i = 0; i < ny; ++i) {
        printf("%.3f ", y[i]);
    }
    if (ny < dim.y) {
        printf("... ");
        for (uint16_t i = dim.y - ny; i < dim.y + 1; ++i) {
            printf("%.3f ", y[i]);
        }
        printf("\n");
    } else {
        printf("... %.3f\n", y[dim.y]);
    }
    printf("Z: \n");
    for (uint16_t i = 0; i < nz; ++i) {
        printf("%.3f ", z[i]);
    }
    if (nz < dim.z) {
        printf("... ");
        for (uint16_t i = dim.z - nz; i < dim.z + 1; ++i) {
            printf("%.3f ", z[i]);
        }
        printf("\n");
    } else {
        printf("... %.3f\n", z[dim.z]);
    }
    mqi::vec3<mqi::ijk_t> idx(0, 0, 0);
    //    mqi::material_id      v = (*node->geo)[idx];
    //    mqi::material_t<R> v = (*node->geo)[idx];
    float v = (*node->geo)[idx];
    printf("From phantom: (%d, %d, %d): %f\n", idx.x, idx.y, idx.z, v);

    for (int i = 0; i < node->n_children; ++i) {
        printf("child node of world: %d, %p\n", i, node->children[i]);
#if defined(__CUDACC__)
        print_node_specification_gpu<R>(node->children[i]);
#else
        print_node_specification<R>(node->children[i]);
#endif
    }
}

#if defined(__CUDACC__)
CUDA_GLOBAL
void
print_density(__half* g_density, uint32_t size_) {
    for (int i = 0; i < size_; i++) {
        printf("g_density %d %.10f\n", i, g_density[i]);
    }
}

CUDA_GLOBAL
void
print_density(float* g_density, uint32_t size_) {
    for (int i = 0; i < size_; i++) {
        printf("g_density %d %.10f\n", i, g_density[i]);
    }
}
#endif

}   // namespace mc

#endif   // PRINT_DATA_CPP
