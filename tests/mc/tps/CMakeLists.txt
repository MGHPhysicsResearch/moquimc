project(tps)

add_executable(tps_env tps_env.cpp)
if (USE_CUDA)
    set_property(TARGET tps_env PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_source_files_properties(tps_env.cpp PROPERTIES LANGUAGE CUDA)
endif ()

target_link_libraries(tps_env PUBLIC
        moqui
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

add_test(
    NAME tps_env
    COMMAND ./tps_env ${PROJECT_SOURCE_DIR}/input_example.in
)
