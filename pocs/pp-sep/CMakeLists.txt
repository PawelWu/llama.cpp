add_executable(pp-sep main.cpp)
target_link_libraries(pp-sep PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
