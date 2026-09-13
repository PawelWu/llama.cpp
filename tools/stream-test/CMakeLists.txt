# stream-test: minimal double-buffering (H2D prefetch + compute overlap) test

set(TARGET llama-stream-test)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE ggml-vulkan ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
