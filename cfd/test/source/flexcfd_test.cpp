#include <string>

#include "flexcfd/flexcfd.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Name is flexcfd", "[library]")
{
  auto const exported = exported_class {};
  REQUIRE(std::string("flexcfd") == exported.name());
}
