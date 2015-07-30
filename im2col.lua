local templet = require 'templet'

local im2col_src = templet.loadstring[[
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
extern "C"
__global__ void im2col_kernel(const float* data_im, float* data_col) {
  CUDA_KERNEL_LOOP(index, ${n}) {
    int w_out = index % ${width_col};
    index /= ${width_col};
    int h_out = index % ${height_col};
    int channel_in = index / ${height_col};
    int channel_out = channel_in * ${ksize_h} * ${ksize_w};
    int h_in = h_out * ${stride_h} - ${pad_h};
    int w_in = w_out * ${stride_w} - ${pad_w};
    data_col += (channel_out * ${height_col} + h_out) * ${width_col} + w_out;
    data_im += (channel_in * ${height} + h_in) * ${width} + w_in;
    for (int i = 0; i < ${ksize_h}; ++i) {
      for (int j = 0; j < ${ksize_w}; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < ${height} && w < ${width}) ?
          data_im[i * ${width} + j] : 0;
        data_col += ${height_col} * ${width_col};
      }
    }
  }
}
]]

local col2im_src = templet.loadstring[[
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

extern "C"
__global__ void col2im_kernel(const float* data_col, float* data_im) {
  CUDA_KERNEL_LOOP(index, ${n}) {
    float val = 0;
    int w = index % {width} + {pad_w};
    int h = (index / {width}) % {height} + {pad_h};
    int c = index / ({width} * {height});
    // compute the start and end of the output
    int w_col_start = (w < {patch_w}) ? 0 : (w - patch_w) / {stride_w} + 1;
    int w_col_end = min(w / {stride_w} + 1, {width_col});
    int h_col_start = (h < {patch_h}) ? 0 : (h - patch_h) / {stride_h} + 1;
    int h_col_end = min(h / {stride_h} + 1, {height_col});
    /*
       for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
    // the col location: [c * {width} * {height} + h_out, w_out]
    int c_col = c * {patch_h} * {patch_w} + (h - h_col * {stride_h}) * ksize + (w - w_col * {stride_w});
    val += data_col[(c_col * {height_col} + h_col) * {width_col} + w_col];
    }
    }
     */
    // equivalent implementation
    int offset = (c * {patch_h} * {patch_w} + h * patch_w + w) * {height_col} * {width_col};
    int coeff_h_col = (1 - {stride_h} * {patch_w} * {height_col}) * {width_col};
    int coeff_w_col = (1 - {stride_w} * {height_col} * {width_col});
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}
]]

local CUDA_NUM_THREADS = 1024

local function GET_BLOCKS(N)
  return math.floor((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
end

function torch.CudaTensor.nn.im2col(dst,src,kW,kH,dW,dH,padW,padH)
  assert(src:nDimension() == 3)
  local height = src:size(2)
  local width = src:size(3)
  local nInputPlane = src:size(1)
  local height_col = (height + 2 * padH - kH) / dH + 1
  local width_col = (width + 2 * padW - kW) / dW + 1
  local n = nInputPlane * height_col * width_col

  dst:resize(nInputPlane*kW*kH, height_col*width_col)

  local kernel = im2col_src{
    height_col = height_col,
    width_col = width_col,
    n = n,
    height = height,
    width = width,
    ksize_h = kH,
    ksize_w = kW,
    pad_h = padH,
    pad_w = padW,
    stride_h = dH,
    stride_w = dW,
    channels = nInputPlane
  }

  -- TODO: cache
  im2col_ptx = im2col_ptx or nvrtc.compileReturnPTX(kernel)
  cutorch.launchPTX(ptx, 'im2col_kernel', {src, dst}, {GET_BLOCKS(n)}, {CUDA_NUM_THREADS})
end

function torch.CudaTensor.nn.col2im(dst,src,kW,kH,dW,dH,padW,padH)
  assert(src:nDimension() == 3)
  local height = src:size(2)
  local width = src:size(3)
  local nInputPlane = src:size(1)
  local height_col = (height + 2 * padH - kH) / dH + 1
  local width_col = (width + 2 * padW - kW) / dW + 1
  local n = nInputPlane * height_col * width_col

  dst:resize(nInputPlane*kW*kH, height_col*width_col)

  local kernel = col2im_src{
    height_col = height_col,
    width_col = width_col,
    n = n,
    height = height,
    width = width,
    ksize_h = kH,
    ksize_w = kW,
    pad_h = padH,
    pad_w = padW,
    stride_h = dH,
    stride_w = dW,
    channels = nInputPlane
  }

  -- TODO: cache
  col2im_ptx = col2im_ptx or nvrtc.compileReturnPTX(kernel)
  cutorch.launchPTX(ptx, 'col2im_kernel', {src, dst}, {GET_BLOCKS(n)}, {CUDA_NUM_THREADS})
end
