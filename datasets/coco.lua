--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/README.md
--
-- Data loading for semantic segmentation on the COCO dataset
--

local ffi = require 'ffi'
local t = require 'datasets/transforms'

local M = {}
local CocoDataset = torch.class('CocoDataset', M)

function CocoDataset:__init(imageInfo, opt, split)
   self.split = split
   assert(imageInfo[split], 'invalid split: ' .. split)
   self.imageInfo = imageInfo[split]
   self.meanstd = {
      mean = imageInfo.mean, -- mean on train set
      std = {1.0, 1.0, 1.0}, -- don't normalize std
   }
   self.__classWeights = imageInfo.classWeights
end

function CocoDataset:get(i)
   local imgPath = ffi.string(self.imageInfo.imgPath[i]:data())
   local annPath = ffi.string(self.imageInfo.annPath[i]:data())

   -- reading an image directly into float will normalize into range 0 to 1
   local img = image.load(imgPath, 3, 'byte'):float()
   local ann = image.load(annPath, 1, 'byte'):squeeze(1):float()

   return {
      input = img,
      target = ann,
   }
end

function CocoDataset:size()
   return self.imageInfo.imgPath:size(1)
end

function CocoDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(self.meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(self.meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

function CocoDataset:classWeights()
   return self.__classWeights
end

return M.CocoDataset
