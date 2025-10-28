add_library(JsonToProtoConverter JsonToProtoConverter.cpp)

add_executable(substrait_runner 
test.cc
JsonToProtoConverter.cpp)
target_link_libraries(substrait_runner PRIVATE
  velox_exec
  velox_substrait_plan_converter
  velox_exec_test_lib
  velox_vector
  velox_core
  velox_memory
  velox_type
  velox_dwio_common
  protobuf::libprotobuf
  velox_dwio_parquet_reader
  velox_dwio_parquet_writer
  ${VELOX_EXTRA_LINK_DEPS} # optional: for fmt, folly, gflags, etc.
)
