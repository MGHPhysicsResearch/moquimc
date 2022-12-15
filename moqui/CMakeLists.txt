add_library(moqui INTERFACE)
target_include_directories(moqui INTERFACE ../)
file(GLOB header_files CONFIGURE_DEPENDS
        "*/*.hpp"
        "*/*/*.hpp"
)
target_sources(moqui PRIVATE ${header_files})
