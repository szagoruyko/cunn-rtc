local templet = require 'templet'
local nvrtc = require 'nvrtc'

local cache = {im2col = {}, col2im = {}}
local unfolds = {}

local im2col_src = templet.loadstring[[
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
extern "C"
__global__ void im2col_kernel(const ${Dtype}* data_im, ${Dtype}* data_col) {
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
    #pragma unroll
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
__global__ void col2im_kernel(const ${Dtype}* data_col, ${Dtype}* data_im) {
  CUDA_KERNEL_LOOP(index, ${n}) {
    ${Dtype} val = 0;
    int w = index % ${width} + ${pad_w};
    int h = (index / ${width}) % ${height} + ${pad_h};
    int c = index / (${width} * ${height});
    // compute the start and end of the output
    int w_col_start = (w < ${patch_w}) ? 0 : (w - ${patch_w}) / ${stride_w} + 1;
    int w_col_end = min(w / ${stride_w} + 1, ${width_col});
    int h_col_start = (h < ${patch_h}) ? 0 : (h - ${patch_h}) / ${stride_h} + 1;
    int h_col_end = min(h / ${stride_h} + 1, ${height_col});

    // equivalent implementation
    int offset = (c * ${patch_h} * ${patch_w} + h * ${patch_w} + w) * ${height_col} * ${width_col};
    int coeff_h_col = (1 - ${stride_h} * ${patch_w} * ${height_col}) * ${width_col};
    int coeff_w_col = (1 - ${stride_w} * ${height_col} * ${width_col});
    #pragma unroll
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}
]]

local function hash(t)
  local s = ''
  for i,v in pairs(t) do
     if type(v) == 'string' then
        s = s..',v'
     else
        s = ('%s,%d'):format(s,v)
     end
  end
  return s
end

local CUDA_NUM_THREADS = 1024

local function GET_BLOCKS(N)
  return math.floor((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
end

local function getDType(t)
   if torch.isTypeOf(t, 'torch.CudaTensor') then
      return 'float'
   elseif torch.isTypeOf(t, 'torch.CudaDoubleTensor') then
      return 'double'
   end
end

function unfolds.im2col(dst,src,kW,kH,dW,dH,padW,padH)
  assert(src:nDimension() == 3)
  local height = src:size(2)
  local width = src:size(3)
  local nInputPlane = src:size(1)
  local height_col = math.floor((height + 2 * padH - kH) / dH) + 1
  local width_col = math.floor((width + 2 * padW - kW) / dW) + 1
  local n = nInputPlane * height_col * width_col

  dst:resize(nInputPlane*kW*kH, height_col*width_col)

  local options = {
     Dtype = getDType(src),
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

  local str = hash(options)
  local ptx = cache.im2col[str]
  if not ptx then
    local kernel = im2col_src(options)
    ptx = nvrtc.compileReturnPTX(kernel)
    cache.im2col[str] = ptx
  end
  cutorch.launchPTX(ptx, 'im2col_kernel', {src, dst}, {GET_BLOCKS(n)}, {CUDA_NUM_THREADS})
  return dst
end

function unfolds.col2im(dst,src,kW,kH,dW,dH,padW,padH)
  assert(src:nDimension() == 2)
  assert(dst:nDimension() == 3)
  local height = dst:size(2)
  local width = dst:size(3)
  local nInputPlane = dst:size(1)
  local height_col = math.floor((height + 2 * padH - kH) / dH) + 1
  local width_col = math.floor((width + 2 * padW - kW) / dW) + 1
  local n = nInputPlane * height * width

  --dst:resize(nInputPlane*kW*kH, height_col*width_col)
  assert(src:size(1) == nInputPlane*kW*kH)
  assert(src:size(2) == height_col*width_col)

  local options = {
     Dtype = getDType(src),
    height_col = height_col,
    width_col = width_col,
    n = n,
    height = height,
    width = width,
    patch_h = kH,
    patch_w = kW,
    pad_h = padH,
    pad_w = padW,
    stride_h = dH,
    stride_w = dW,
    channels = nInputPlane
  }

  local str = hash(options)
  local ptx = cache.col2im[str]
  if not ptx then
    local kernel = col2im_src(options)
    ptx = nvrtc.compileReturnPTX(kernel)
    cache.col2im[str] = ptx
  end
  cutorch.launchPTX(ptx, 'col2im_kernel', {src, dst}, {GET_BLOCKS(n)}, {CUDA_NUM_THREADS})
end

return unfolds
