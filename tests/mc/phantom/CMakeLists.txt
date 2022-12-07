project(phantom)

add_executable(phantom_env phantom_env.cpp)
if (GPU)
    set_property(TARGET phantom_env PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_source_files_properties(phantom_env.cpp PROPERTIES LANGUAGE CUDA)
endif ()

target_include_directories(phantom_env PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${GDCM_INCLUDE_DIRS}
        ../../../
)

target_link_directories(phantom_env PUBLIC ${GDCM_DIR})

target_link_libraries(phantom_env
        ${COREFOUNDATION_LIBRARY}
        gdcmCommon
        gdcmDSED
        gdcmMEXD
        gdcmjpeg12
        gdcmjpeg8
        gdcmDICT
        gdcmIOD
        gdcmMSFF
        gdcmjpeg16
        ZLIB::ZLIB
        )

#https://stackoverflow.com/questions/3065220/ctest-with-multiple-commands
add_test(
    NAME phantom_env
    COMMAND
        bash -c "python3 ${PROJECT_SOURCE_DIR}/create_phantom.py ; \
        ${CMAKE_CURRENT_BINARY_DIR}/phantom_env --lxyz 100 100 350 --pxyz 0.0 0.0 -175 --nxyz 200 200 350 --spot_energy 200.0 0.0 --spot_position 0 0 0.5 --spot_size 30.0 30.0 --histories 100000 --phantom_path /tmp/water_phantom.raw --output_prefix ./ --gpu_id 0"
)
