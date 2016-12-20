--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
--
-- Simple single-threaded data loader
--

local datasets = require 'datasets/init'

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt)
   local loaders = {}
   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end
   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   self.dataset = dataset
   self.preprocess = dataset:preprocess()
   self.__size = dataset:size() -- number of samples
   self.batchSize = opt.batchSize
   self.split = split
end

-- number of batches
function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   -- shuffle training data only
   local perm = torch.randperm(self.__size)
   if self.split == 'val' then perm = torch.range(1, self.__size) end
   local idx = 1 -- sample index
   local n = 0 -- batch index

   local function loop()
      if idx <= self.__size then
         local indices = perm:narrow(1, idx, math.min(self.batchSize, self.__size - idx + 1))
         local sz = indices:size(1) -- actual batch size

         -- build batch
         local batchInputs, batchTargets, inputSize, targetSize
         for i, index in ipairs(indices:totable()) do
            local sample = self.dataset:get(index)
            local input, target = self.preprocess(sample.input, sample.target)
            if not batchInputs then
               inputSize = input:size():totable() -- image size
               targetSize = target:size():totable()
               batchInputs = torch.FloatTensor(sz, table.unpack(inputSize))
               batchTargets = torch.FloatTensor(sz, table.unpack(targetSize))
            end
            batchInputs[i]:copy(input)
            batchTargets[i]:copy(target)
         end
         collectgarbage()

         local sample = {
            input = batchInputs:view(sz, table.unpack(inputSize)),
            target = batchTargets:view(sz, table.unpack(targetSize)),
         }

         n = n + 1
         idx = idx + self.batchSize
         return n, sample
      end
   end

   return loop
end

return M.DataLoader
