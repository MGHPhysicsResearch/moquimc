project(tps)

add_executable(tps_env tps_env.cpp)

if (GPU)
    set_property(TARGET tps_env PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_source_files_properties(tps_env.cpp PROPERTIES LANGUAGE CUDA)
endif ()

target_include_directories(tps_env PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${GDCM_INCLUDE_DIRS}
        ../../../
)

target_link_directories(tps_env PUBLIC ${GDCM_DIR})

target_link_libraries(tps_env
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
