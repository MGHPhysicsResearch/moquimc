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
