--
-- Implements SegNet from the following paper:
--
--    SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
--
-- See following links:
--
--    https://arxiv.org/pdf/1511.00561v3.pdf
--

local loadcaffe = require 'loadcaffe'

local Convolution = cudnn.SpatialConvolution
local SBatchNorm = cudnn.SpatialBatchNormalization
local ReLU = cudnn.ReLU
local MaxPool = nn.SpatialMaxPooling
local MaxUnpool = nn.SpatialMaxUnpooling

local function createModel(opt)
   assert(paths.filep(opt.prototxt), 'prototxt not found: ' .. opt.prototxt)
   assert(paths.filep(opt.caffemodel, 'caffemodel not found: ' .. opt.caffemodel))
   vgg16 = loadcaffe.load(opt.prototxt, opt.caffemodel, opt.backend)

   -- https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/vgg.lua
   local inChannels, outChannels
   local maxPoolLayers = {}
   local maxPoolIdx = 1

   local model = nn.Sequential()
   local encoderCfg = {1, 3, 'M', 6, 8, 'M', 11, 13, 15, 'M', 18, 20, 22, 'M', 25, 27, 29, 'M'}
   inChannels = 3
   for _, l in ipairs(encoderCfg) do
      if l == 'M' then
         local maxPool = MaxPool(2,2,2,2)
         model:add(maxPool)
         maxPoolLayers[maxPoolIdx] = maxPool
         maxPoolIdx = maxPoolIdx + 1
      else
         model:add(vgg16:get(l))
         outChannels = vgg16:get(l).nOutputPlane
         model:add(SBatchNorm(outChannels))
         model:add(ReLU(true))
         inChannels = outChannels
      end
   end
   local decoderCfg = {'M', 512, 512, 512, 'M', 512, 512, 256, 'M', 256, 256, 128, 'M', 128, 64, 'M', 64, opt.nClasses+1}
   for i, l in ipairs(decoderCfg) do
      if l == 'M' then
         maxPoolIdx = maxPoolIdx - 1
         model:add(MaxUnpool(maxPoolLayers[maxPoolIdx]))
      else
         outChannels = l
         model:add(Convolution(inChannels,outChannels,3,3,1,1,1,1))
         if i < 18 then
            model:add(SBatchNorm(outChannels))
            model:add(ReLU(true))
         end
         inChannels = outChannels
      end
   end

   return model
end

return createModel
