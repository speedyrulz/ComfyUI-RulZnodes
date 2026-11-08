/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified for RulZnodes from csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu
 * (SageAttention d1a57a5): launches the block-list kernel for NHD, per-warp, non-causal attention only.
 */
#include <ATen/cuda/CUDAContext.h>
#include "qk_sparse_sm89.cuh"

// NHD int8 q/k (per-warp scales), fp8 v laid out (b, d, h, kv_pad), fp32+fp16 PV accumulation,
// matching sageattn on sm89/sm120. lut: (b, h, ceil(qo_len/128), lut_stride) ascending 64-key block
// ids, lut_count: (b, h, ceil(qo_len/128)) number of valid ids per row, at least 1.
void qk_int8_sv_f8_sparse_attn(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output,
                               torch::Tensor query_scale, torch::Tensor key_scale, torch::Tensor value_scale,
                               torch::Tensor lut, torch::Tensor lut_count, float sm_scale)
{
  CHECK_CUDA(query); CHECK_CUDA(key); CHECK_CUDA(value); CHECK_CUDA(output); CHECK_CUDA(lut); CHECK_CUDA(lut_count);
  CHECK_LASTDIM_CONTIGUOUS(query); CHECK_LASTDIM_CONTIGUOUS(key); CHECK_CONTIGUOUS(value); CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale); CHECK_CONTIGUOUS(key_scale); CHECK_CONTIGUOUS(value_scale);
  CHECK_CONTIGUOUS(lut); CHECK_CONTIGUOUS(lut_count);
  CHECK_DTYPE(query, torch::kInt8); CHECK_DTYPE(key, torch::kInt8);
  CHECK_DTYPE(lut, torch::kInt32); CHECK_DTYPE(lut_count, torch::kInt32);

  const int batch_size = query.size(0);
  const int qo_len = query.size(1);
  const int kv_len = key.size(1);
  const int num_qo_heads = query.size(2);
  const int num_kv_heads = key.size(2);
  const int head_dim = query.size(3);
  constexpr int CTA_Q = 128;
  constexpr int CTA_K = 64;
  constexpr int WARP_Q = 32;
  constexpr int WARP_K = 64;
  TORCH_CHECK(num_qo_heads == num_kv_heads, "grouped kv heads not supported");
  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
  CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);
  CHECK_SHAPE(lut_count, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
  TORCH_CHECK(lut.dim() == 4 && lut.size(0) == batch_size && lut.size(1) == num_qo_heads && lut.size(2) == div_ceil(qo_len, CTA_Q));
  TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K);

  auto output_dtype = output.scalar_type();
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
      size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));
      auto kernel_func = qk_int_sv_f8_sparse_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
                                                         float, true, DTypeOut, ComputeUnit::kCudaCore, MaskMode::kNone, false, true, false, true>;
      cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
      dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
      dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
      kernel_func<<<grid, block, smem_max, at::cuda::getCurrentCUDAStream()>>>(
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), reinterpret_cast<int8_t*>(value.data_ptr()),
        reinterpret_cast<DTypeOut*>(output.data_ptr()), nullptr,
        reinterpret_cast<float*>(query_scale.data_ptr()), reinterpret_cast<float*>(key_scale.data_ptr()),
        reinterpret_cast<float*>(value_scale.data_ptr()), nullptr,
        qo_len, kv_len, 1,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(2), value.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        sm_scale, lut.data_ptr<int32_t>(), lut_count.data_ptr<int32_t>(), lut.size(3));
    });
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_int8_sv_f8_sparse_attn", &qk_int8_sv_f8_sparse_attn);
}
