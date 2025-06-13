# ====================================================================
# C TRANSFORMER - SIMPLIFIED CMAKE CONFIGURATION
# ====================================================================
# Streamlined CMakeLists.txt that builds only in model/releases directory

cmake_minimum_required(VERSION 3.12)
project(Aaryan C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS_DEBUG "-g -Wall -Wextra -O0 -DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra -DNDEBUG")
    if(WIN32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc")
    endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_C_FLAGS_DEBUG "/Od /W3 /Zi /DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /W3 /DNDEBUG")
endif()

# Platform-specific definitions
if(WIN32)
    add_definitions(-DPLATFORM_WINDOWS)
    set(PLATFORM "windows-x64")
    set(EXE_SUFFIX ".exe")
else()
    add_definitions(-DPLATFORM_LINUX)
    if(APPLE)
        set(PLATFORM "macos-x64")
    else()
        set(PLATFORM "linux-x64")
    endif()
    set(EXE_SUFFIX "")
endif()

# Include directories
include_directories(include)

# Source files
set(SOURCES
    main.c
    src/matrix.c
    src/optimizer.c
    src/platform.c
    src/tokenizer.c
    src/transformer_model.c
    src/utils.c
    src/interactive_cli.c
)

# Set output directories to model/releases structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/model/releases/${PLATFORM}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/model/releases/${PLATFORM}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/model/releases/${PLATFORM}/bin)

# Find OpenMP (optional)
find_package(OpenMP)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Main executable
add_executable(Aaryan ${SOURCES})
target_link_libraries(Aaryan m)
if(OpenMP_C_FOUND)
    target_link_libraries(Aaryan OpenMP::OpenMP_C)
endif()

# Preprocessor executable  
add_executable(preprocess_csv preprocess_csv.c src/utils.c)
target_link_libraries(preprocess_csv m)

# Windows-specific linking
if(WIN32)
    target_link_libraries(Aaryan ws2_32)
    target_link_libraries(preprocess_csv ws2_32)
endif()

# Create release directory structure
add_custom_target(release_structure ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/model/releases/${PLATFORM}/bin"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/model/releases/${PLATFORM}/data"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/model/releases/${PLATFORM}/models"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/model/builds/${PLATFORM}-obj"
    COMMENT "Creating release directory structure"
)

# Copy data files to release
add_custom_command(TARGET release_structure POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/data" "${CMAKE_SOURCE_DIR}/model/releases/${PLATFORM}/data"
    COMMENT "Copying data files to release directory"
)

# Make executables depend on directory structure
add_dependencies(Aaryan release_structure)
add_dependencies(preprocess_csv release_structure)
