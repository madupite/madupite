# python/CMakeLists.txt

nanobind_add_module(
    pymadupite
    NB_STATIC
    pymadupite.cpp
)

target_link_libraries(pymadupite PRIVATE madupite)

set_target_properties(pymadupite PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_madupite"
)

# Install the module
install(TARGETS pymadupite
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/madupite"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/madupite")

# Copy the module after build into the madupite directory and rename to _madupite.so
add_custom_command(TARGET pymadupite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pymadupite>
        "${CMAKE_SOURCE_DIR}/madupite/_madupite.so"
    COMMENT "Copying module to madupite directory"
)
