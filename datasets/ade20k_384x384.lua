--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/README.md
--
-- Data loading for semantic segmentation on the ADE20K dataset, preprocessed into hdf5 format
--

local hdf5 = require 'hdf5'
local t = require 'datasets/transforms'

local M = {}
local Ade20kDataset = torch.class('Ade20kDataset', M)

function Ade20kDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], 'invalid split: ' .. split)
   self.indices = imageInfo[split].indices
   assert(paths.filep(imageInfo.h5Path),
          'h5 file not found: ' .. imageInfo.h5Path)
   self.h5File = hdf5.open(imageInfo.h5Path, 'r')
   self.nChannels = imageInfo.nChannels
   self.imageSize = imageInfo.imageSize
   self.split = split
   self.meanstd = {
      mean = imageInfo.mean,
      std = {1.0, 1.0, 1.0}, -- don't normalize std
   }
end

function Ade20kDataset:get(i)
   local image = self.h5File:read('/images'):partial(
                    self.indices[i], {1, self.nChannels},
                    {1, self.imageSize}, {1, self.imageSize}):squeeze(1):float()
   local annotation = self.h5File:read('/annotations'):partial(self.indices[i],
                         {1, self.imageSize}, {1, self.imageSize}):squeeze(1):float()

   return {
      input = image,
      target = annotation,
   }
end

-- number of samples
function Ade20kDataset:size()
   return self.indices:size(1)
end

function Ade20kDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(self.meanstd), -- mean subtraction
         t.HorizontalFlip(0.5), -- flip with probability 0.5
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(self.meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.Ade20kDataset
