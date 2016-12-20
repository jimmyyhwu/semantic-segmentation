--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/README.md
--
-- Setup code for semantic segmentation on the ADE20K dataset, preprocessed into hdf5 format
--

local hdf5 = require 'hdf5'
local utils = require 'utils'

local M = {}

local function readSplits(jsonPath)
   info = utils.readJson(jsonPath)

   local trainIndices = {} -- indices into hdf5 data
   local valIndices = {}
   for i, imageInfo in ipairs(info.images) do
      if imageInfo.split == 'train' then
         table.insert(trainIndices, i)
      elseif imageInfo.split == 'val' then
         table.insert(valIndices, i)
      else
         error('invalid split: ' .. imageInfo.split)
      end
   end
   
   -- convert to tensors for faster loading
   local trainIndex = torch.LongTensor()
   local valIndex = torch.LongTensor()
   trainIndex = torch.LongTensor(trainIndices)
   valIndex = torch.LongTensor(valIndices)

   return trainIndex, valIndex
end

function M.exec(opt, cacheFile)
   print('=> Setting up ADE20K preprocessed hdf5 for semantic segmentation')

   local jsonPath = paths.concat(opt.data, 'ade20k_384x384.json')
   local h5Path = paths.concat(opt.data, 'ade20k_384x384.h5')
   assert(paths.filep(jsonPath), 'json file not found: ' .. jsonPath)
   assert(paths.filep(h5Path), 'h5 file not found: ' .. h5Path)

   print(' | reading train/val splits from json')
   local trainIndex, valIndex = readSplits(jsonPath)

   print(' | performing safety checks on h5 file')
   local h5File = hdf5.open(h5Path, 'r')
   local imagesSize = h5File:read('/images'):dataspaceSize()
   assert(#imagesSize == 4, '/images should be 4D tensor')
   assert(imagesSize[3] == imagesSize[4], 'image width and height must match')
   local annotationsSize = h5File:read('/annotations'):dataspaceSize()
   assert(#annotationsSize == 3, '/annotations should be 3D tensor')
   assert(annotationsSize[1] == imagesSize[1],
          '/images count should equal /annotations count')

   print(' | computing mean of dataset')
   local nImages = imagesSize[1]
   local tt = h5File:read('/images'):partial({1, nImages}, {1, imagesSize[2]},
                                             {1, 384}, {1, 384}):float()
   tt = tt:transpose(2,4)
   tt = tt:reshape(nImages*384*384, 3)
   local mean = tt:mean(1):squeeze():float()

   local info = {
      basedir = opt.data,
      h5Path = h5Path,
      nChannels = imagesSize[2],
      imageSize = imagesSize[3],
      mean = mean,
      train = {
         indices = trainIndex,
      },
      val = {
         indices = valIndex,
      },
   }

   print(' | saving image info to ' .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
