#pragma once

#include <cstdint>

// Round up to the nearest multiple of 64.
//
// Emitted wrapper code calls this by name, and a translation unit can pull in
// both the JIT and the AOTI prelude, so it needs a single definition.
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
