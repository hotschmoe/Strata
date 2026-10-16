#pragma once

namespace strata::kernels {
// SYCL head-only experiment. Default OFF; other dispatch remains native_mmvq.
void native_mmvq_head(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);
} // namespace strata::kernels
