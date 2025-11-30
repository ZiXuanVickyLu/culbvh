# Platform-specific compiler and tool settings
if(WIN32)
  # Windows - Let CMake auto-detect MSVC from Visual Studio environment
  # Don't force compilers; use Visual Studio generator or run from VS Developer Command Prompt
  # This ensures all Windows SDK tools (mt.exe, rc.exe) are properly found
  message(STATUS "Windows detected - using auto-detected compiler (MSVC recommended for CUDA)")

    
elseif(UNIX AND NOT APPLE)
    # Linux and other Unix systems
    find_program(CLANG_C_COMPILER NAMES clang clang-20 clang-19 clang-18 clang-17 clang-16 clang-15)
    find_program(CLANG_CXX_COMPILER NAMES clang++ clang++-20 clang++-19 clang++-18 clang++-17 clang++-16 clang++-15)

    if(CLANG_C_COMPILER)
        set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}" CACHE STRING "C compiler" FORCE)
        message(STATUS "Found Clang C compiler: ${CLANG_C_COMPILER}")
    else()
        message(WARNING "Clang not found, falling back to default")
        set(CMAKE_C_COMPILER "clang" CACHE STRING "C compiler" FORCE)
    endif()

    if(CLANG_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}" CACHE STRING "C++ compiler" FORCE)
        message(STATUS "Found Clang C++ compiler: ${CLANG_CXX_COMPILER}")
    else()
        message(WARNING "Clang++ not found, falling back to default")
        set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler" FORCE)
    endif()

    find_program(NINJA_PROGRAM NAMES ninja ninja-build)
    if(NINJA_PROGRAM)
        set(CMAKE_MAKE_PROGRAM "${NINJA_PROGRAM}" CACHE STRING "Build tool" FORCE)
    else()
        message(WARNING "Ninja not found, using default")
        set(CMAKE_MAKE_PROGRAM "ninja" CACHE STRING "Build tool" FORCE)
    endif()

else()
    # Fallback for other systems
    message(WARNING "Unknown platform, using default clang settings")
    set(CMAKE_C_COMPILER "clang" CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler" FORCE)
    set(CMAKE_MAKE_PROGRAM "ninja" CACHE STRING "Build tool" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
  # MSVC warning flags
  set(CMAKE_C_FLAGS_INIT "/W3")
  set(CMAKE_CXX_FLAGS_INIT "/W3")
else()
  # Clang/GCC warning flags
  set(CMAKE_C_FLAGS_INIT "-Wall")
  set(CMAKE_CXX_FLAGS_INIT "-Wall")
endif()

message(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Using build tool: ${CMAKE_MAKE_PROGRAM}")