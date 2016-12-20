--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
--
-- Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

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
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = opt.batchSize
   self.split = split
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   -- shuffle training data only
   local perm = torch.randperm(size)
   if self.split == 'val' then perm = torch.range(1, size) end

   local index, sample = 1, nil
   local function enqueue()
      while index <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, index, math.min(batchSize, size - index + 1))
         threads:addjob(
            function(indices)
               local sz = indices:size(1)
               local batchInputs, batchTargets, inputSize, targetSize
               for i, index in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(index)
                  local input, target = _G.preprocess(sample.input, sample.target)
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
               return {
                  input = batchInputs:view(sz, table.unpack(inputSize)),
                  target = batchTargets:view(sz, table.unpack(targetSize)),
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices
         )
         index = index + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
