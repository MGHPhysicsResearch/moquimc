add_library(moqui INTERFACE)
target_include_directories(moqui INTERFACE ../)
file(GLOB header_files CONFIGURE_DEPENDS
        "*/*.hpp"
        "*/*/*.hpp"
)
target_sources(moqui PRIVATE ${header_files})
target_link_libraries(moqui INTERFACE
    gdcmCommon
    gdcmDSED
    gdcmMEXD
    gdcmjpeg12
    gdcmjpeg8
    gdcmDICT
    gdcmIOD
    gdcmMSFF
    gdcmjpeg16
)
