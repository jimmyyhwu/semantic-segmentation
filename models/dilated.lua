--
-- Implements front end from the following paper:
--
--    Multi-Scale Context Aggregation by Dilated Convolutions
--
-- See following links:
--
--    https://arxiv.org/pdf/1511.07122v3.pdf
--    https://github.com/fyu/dilation
--

local loadcaffe = require 'loadcaffe'

local Convolution = cudnn.SpatialConvolution
local DilatedConv = nn.SpatialDilatedConvolution

local function createModel(opt)
   assert(paths.filep(opt.prototxt), 'prototxt not found: ' .. opt.prototxt)
   assert(paths.filep(opt.caffemodel), 'caffemodel not found: ' .. opt.caffemodel)
   local vgg16 = loadcaffe.load(opt.prototxt, opt.caffemodel, opt.backend)

   -- remove pool4, pool5, and torch_view
   vgg16:remove(24)
   vgg16:remove(30)
   vgg16:remove(30)

   local model = nn.Sequential()
   for i = 1, 36 do
      local layer = vgg16:get(i)
      if i == 1 then
         -- bgr to rgb
         layer.weight = layer.weight:index(2, torch.LongTensor{3,2,1})
         model:add(layer)
      elseif i == 24 or i == 26 or i == 28 then
         local dilatedLayer = DilatedConv(512,512,3,3,1,1,2,2,2,2)
         dilatedLayer.weight:copy(layer.weight)
         model:add(dilatedLayer)
      elseif i == 30 then
         model:add(DilatedConv(512,4096,7,7,1,1,12,12,4,4))
      elseif i == 33 then
         model:add(Convolution(4096,4096,1,1))
      elseif i == 36 then
         model:add(Convolution(4096,opt.nClasses+1,1,1))
      else
         model:add(layer)
      end
   end
   model:add(nn.SpatialUpSamplingBilinear(8))
   return model
end

return createModel
