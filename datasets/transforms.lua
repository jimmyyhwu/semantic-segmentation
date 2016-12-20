--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua
--

require 'image'

local M = {}

function M.Compose(transforms)
   return function(input, target)
      for _, transform in ipairs(transforms) do
         input, target = transform(input, target)
      end
      return input, target
   end
end

function M.ColorNormalize(meanstd)
   return function(img, anno)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img, anno
   end
end

function M.HorizontalFlip(prob)
   return function(input, target)
      if torch.uniform() < prob then
         input = image.hflip(input)
         target = image.hflip(target)
      end
      return input, target
   end
end

return M
