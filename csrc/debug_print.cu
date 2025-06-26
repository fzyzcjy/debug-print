#include <c10/cuda/CUDAStream.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define AT_DISPATCH_CASE_FLOATING_AND_REDUCED_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                        \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(TYPE, NAME, ...)       \
  AT_DISPATCH_SWITCH(                                                          \
      TYPE, NAME,                                                              \
      AT_DISPATCH_CASE_FLOATING_AND_REDUCED_FLOATING_TYPES(__VA_ARGS__))

__device__ void PrintCommon(void* x, const char* name_ptr, const bool print_ptr, const bool print_shape) {
  if (name_ptr != nullptr) {
    printf("name: %s\n", name_ptr);
  }
  if (print_ptr) {
    printf("addr: %lld\n", x);
  }
}

template <typename scalar_t>
__device__ void PrintElem(scalar_t value) {
    if constexpr (std::is_floating_point<scalar_t>::value) {
      printf("%.4f, ", float(x[i * stride_0]));
    } else if constexpr (std::is_integral<scalar_t>::value) {
      printf("%lld, ", static_cast<long long>(x[i * stride_0]));
    } else {
      printf("?, ");
    }
}

template <typename float_t>
__global__ void PrintTensor1D(
    float_t *__restrict__ x,
    const size_t shape_0,
    const size_t stride_0,
    const char* name_ptr, const bool print_ptr, const bool print_shape
) {
  PrintCommon(x, name_ptr, print_ptr);
  if (print_shape) {
    printf("shape=(%d), stride=(%d)", (int) shape_0, (int) stride_0);
  }
  printf("[");
  for (size_t index_0 = 0; index_0 < shape_0; ++index_0) {
    PrintElem(x[index_0 * stride_0]);
  }
  printf("]");
  printf("\n");
}

template <typename float_t>
__global__ void PrintTensor2D(
    float_t *__restrict__ x,
    const size_t shape_0, const size_t shape_1,
    const size_t stride_0, const size_t stride_1,
    const char* name_ptr, const bool print_ptr, const bool print_shape
) {
  PrintCommon(x, name_ptr, print_ptr);
  if (print_shape) {
    printf("shape=(%d, %d), stride=(%d, %d)", (int) shape_0, (int) shape_1, (int) stride_0, (int) stride_1);
  }
  printf("[");
  for (size_t index_0 = 0; index_0 < shape_0; ++index_0) {
    printf("[");
    for (size_t index_1 = 0; index_1 < shape_1; ++index_1) {
      PrintElem(x[index_0 * stride_0 + index_1 * stride_1]);
    }
    printf("]");
  }
  printf("]");
  printf("\n");
}

template <typename float_t>
__global__ void PrintTensor3D(
    float_t *__restrict__ x,
    const size_t shape_0, const size_t shape_1, const size_t shape_2,
    const size_t stride_0, const size_t stride_1, const size_t stride_2,
    const char* name_ptr, const bool print_ptr, const bool print_shape
) {
  PrintCommon(x, name_ptr, print_ptr);
  if (print_shape) {
    printf("shape=(%d, %d, %d), stride=(%d, %d, %d)", (int) shape_0, (int) shape_1, (int) shape_2, (int) stride_0, (int) stride_1, (int) stride_2);
  }
  printf("[");
  for (size_t index_0 = 0; index_0 < shape_0; ++index_0) {
    printf("[");
    for (size_t index_1 = 0; index_1 < shape_1; ++index_1) {
      printf("[");
      for (size_t index_2 = 0; index_2 < shape_2; ++index_2) {
        PrintElem(x[index_0 * stride_0 + index_1 * stride_1]);
      }
      printf("]");
    }
    printf("]");
  }
  printf("]");
  printf("\n");
}

void PrintTensor(torch::Tensor x, std::optional<torch::Tensor> name_buffer, bool print_ptr, bool print_shape) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index());
  TORCH_CHECK(x.is_cuda(), "The input tensor should be a CUDA tensor");

  const char* name_ptr = name_buffer.has_value() ? reinterpret_cast<char*>(name_buffer->data_ptr<uint8_t>()) : nullptr;

  if (x.is_floating_point()) {
    if (x.dim() == 1) {
      AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(
          x.scalar_type(), "PrintTensor1D", ([&] {
            PrintTensor1D<<<1, 1, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                x.size(0), x.stride(0),
                name_ptr, print_ptr, print_shape
            );
          }));
    } else if (x.dim() == 2) {
      AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(
          x.scalar_type(), "PrintTensor2D", ([&] {
            PrintTensor2D<<<1, 1, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                x.size(0), x.size(1), x.stride(0), x.stride(1),
                name_ptr, print_ptr, print_shape
            );
          }));
    } else if (x.dim() == 3) {
      AT_DISPATCH_FLOATING_AND_REDUCED_FLOATING_TYPES(
          x.scalar_type(), "PrintTensor3D", ([&] {
            PrintTensor3D<<<1, 1, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                x.size(0), x.size(1), x.size(2), x.stride(0), x.stride(1), x.stride(2),
                name_ptr, print_ptr, print_shape
            );
          }));
    } else {
      // NOTE(Zihao): I'm just too lazy to do this, codegen for higher
      // dimensions should be a better idea
      TORCH_CHECK(false, "Input dimension not supported.");
    }
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess,
                "PrintTensor failed with error " +
                    std::string(cudaGetErrorString(status)));
  } else {
    if (x.dim() == 1) {
      AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "PrintTensor1D", ([&] {
           PrintTensor1D<<<1, 1, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                x.size(0), x.stride(0),
                name_ptr, print_ptr, print_shape
            );
      }));
    } else if (x.dim() == 2) {
      AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "PrintTensor2D", ([&] {
       PrintTensor2D<<<1, 1, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                x.size(0), x.size(1), x.stride(0), x.stride(1),
                name_ptr, print_ptr, print_shape
            );
     }));
    } else if (x.dim() == 3) {
      AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "PrintTensor3D", ([&] {
       PrintTensor3D<<<1, 1, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                x.size(0), x.size(1), x.size(2), x.stride(0), x.stride(1), x.stride(2),
                name_ptr, print_ptr, print_shape
            );
     }));
    } else {
      // NOTE(Zihao): I'm just too lazy to do this, codegen for higher
      // dimensions should be a better idea
      TORCH_CHECK(false, "Input dimension not supported.");
    }
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess,
                "PrintTensor failed with error " +
                    std::string(cudaGetErrorString(status)));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_tensor", &PrintTensor,
        "Print tensor inside cuda kenrels for debugging CUDAGraph");
}
