#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>

#include "cusparse.h"

#include <pybind11/pybind11.h>

#include <THC/THCGeneral.hpp>

#include <torch/extension.h>

namespace py = pybind11;

using namespace at::sparse;

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

#define CHECK_ERROR(str) \
    {cudaDeviceSynchronize(); cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout);}}


at::Tensor expand_values_if_needed(const at::Tensor& values) {
    // expand
    if (values.dim() == 0) {
        // Mimic Numpy behavior here and treat it as a 1D tensor
        return values.expand({1});
    } else {
        return values;
    }
}

at::Tensor sparse_coo_tensor_gpu(const at::Tensor& indices, 
                                    const at::Tensor& values_, 
                                    at::ArrayRef<int64_t> size) {

    at::Tensor values = expand_values_if_needed(values_); 

    int64_t sparse_dim = indices.size(0);
    int64_t dense_dim = values.dim() - 1;

    return at::_sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, size, indices, values, values.options().layout(at::kSparse));
}

template<typename T>
void printCusparseDnMat(int64_t rows, int64_t cols, int64_t ld, T *values_dev) {
  T* values_host = new T[rows*cols];
  cudaMemcpy(values_host, values_dev, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
  for (int64_t row = 0; row < rows; row++) {
    for (int64_t col = 0; col < cols; col++) {
      // Cusparse dense matrices are stored in column-major order
      std::cout << values_host[col*rows+row] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < rows*cols; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
}

template<typename T>
void printCusparseSpMat(int32_t rows, int32_t cols, int32_t nnz, int32_t *row_indices_dev,
                            int32_t *col_indices_dev, T *values_dev) {
  T* values_host = new T[nnz];
  int32_t* row_indices_host = new int32_t[nnz];
  int32_t* col_indices_host = new int32_t[nnz];
  cudaMemcpy(values_host, values_dev, nnz*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_indices_host, row_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(col_indices_host, col_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);

  for (int64_t i = 0; i < nnz; i++) {
    std::cout << "(" << row_indices_host[i]
      << ", " << col_indices_host[i]
      << "): " << values_host[i] << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  row_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << row_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  col_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << col_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
  delete [] row_indices_host;
  delete [] col_indices_host;
}

template<typename T> 
void csrmm2(
  cusparseHandle_t handle,
  cusparseOperation_t opa, cusparseOperation_t opb, 
  int32_t m, int32_t n, int32_t k, int32_t nnz, 
  T alpha, T *csrvala, int *csrrowptra, int *csrcolinda, 
  T *b, int32_t ldb, T beta, T *c, int32_t ldc)
{
  static_assert(std::is_same<float, T>::value); 
  constexpr auto cusparse_value_type = CUDA_R_32F; 

  if (csrvala == nullptr || b == nullptr || c == nullptr) return; 

  int32_t ma = m, ka = k; 

  cusparseSpMatDescr_t descA; 
  cusparseCreateCsr(
    &descA,                     /* output */
    ma, ka, nnz,                /* rows, cols, number of non zero elements */
    csrrowptra,                 /* row offsets of the sparse matrix, size = rows +1 */
    csrcolinda,                 /* column indices of the sparse matrix, size = nnz */
    csrvala,                    /* values of the sparse matrix, size = nnz */
    CUSPARSE_INDEX_32I,         /* data type of row offsets index */
    CUSPARSE_INDEX_32I,         /* data type of col indices */
    CUSPARSE_INDEX_BASE_ZERO,   /* base index of row offset and col indes */
    cusparse_value_type         /* data type of values */
  ); 

  int32_t kb = k, nb = n;

  cusparseDnMatDescr_t descB; 
  cusparseCreateDnMat(
    &descB,               /* output */
    kb, nb, ldb,          /* rows, cols, leading dimension */
    b,                    /* values */
    cusparse_value_type,  /* data type of values */
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ); 

  cusparseDnMatDescr_t descC; 
  cusparseCreateDnMat(
    &descC,               /* output */
    m, n, ldc,            /* rows, cols, leading dimension */
    c,                    /* values */ 
    cusparse_value_type,  /* data type of values */ 
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ); 

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize; 
  cusparseSpMM_bufferSize(
    handle, opa, opb,     
    &alpha,               
    descA, descB, 
    &beta, 
    descC, 
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_CSRMM_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    &bufferSize           /* output */
  ); 

  void* externalBuffer; // device pointer
  cudaMalloc(&externalBuffer, bufferSize); 

  cusparseSpMM(
    handle, opa, opb, 
    &alpha, 
    descA, descB, 
    &beta, 
    descC, 
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_SPMM_CSR_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    externalBuffer        /* external buffer */
  ); 

  cudaFree(externalBuffer); 
  cusparseDestroySpMat(descA); 
  cusparseDestroyDnMat(descB); 
  cusparseDestroyDnMat(descC); 

  cudaDeviceSynchronize();
  cusparseDestroy(handle); 
}


// at::Tensor spmm_gpu(const at::Tensor& A_rowindices, 
void spmm_gpu(const at::Tensor& A_rowindices, 
                        const at::Tensor& A_colindices,
                        const at::Tensor& A_values, 
                        int32_t n,
                        int32_t m,
                        at::Tensor& B,
                        at::Tensor& C) {

    // cusparseHandle_t handle;
    // CHECK_CUSPARSE(cusparseCreate(&handle));
    auto state = at::globalContext().lazyInitCUDA();
    // auto handle = THCState_getCurrentSparseHandle(state);
    cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();

    // Impl1 -- coo2csr + csrmm2
    int nnz = A_values.size(0);

    clock_t start, stop;
    
    int32_t *d_a_csrrows;

    
    // int devid_old = 0;
    // cudaGetDevice(&devid_old);
    // cudaSetDevice(devid);

    cudaMalloc(&d_a_csrrows, (n + 1) * sizeof(int32_t));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        A_rowindices.data<int>(), 
                                        nnz, 
                                        n, 
                                        d_a_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));

    float alpha = 1;
    float beta = 1;

    // auto C = torch::ones({n, B.size(1)}, torch::dtype(torch::kDouble).device(torch::kCUDA));

    int32_t b_row = B.size(0);
    int32_t b_col = B.size(1);
    int32_t c_row = C.size(0);
    int32_t c_col = C.size(1);
    
    // Row-major to column-major
    // B.t_();
    // B.set_data(B.contiguous());
    // B.set_data(B.view({b_row, b_col}));
    C.t_();
    C.set_data(C.contiguous());
    C.set_data(C.view({c_row, c_col}));

    csrmm2(handle,
           CUSPARSE_OPERATION_NON_TRANSPOSE,
           CUSPARSE_OPERATION_TRANSPOSE,
           n,
           b_col,
           m,
           nnz,
           alpha,
           A_values.data<float>(),
           d_a_csrrows,
           A_colindices.data<int>(),
           B.data<float>(),
           B.size(1),
           beta,
           C.data<float>(),
           n); 

    cudaFree(d_a_csrrows);

    // Column-major to row-major
    // B.set_data(B.view({b_col, b_row}));
    // B.t_();
    C.set_data(C.view({c_col, c_row}));
    C.t_();
}


 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse Tensor GPU-only constructor");
    m.def("spmm_gpu", &spmm_gpu, "SpMM wrapper for cusparse");
}
