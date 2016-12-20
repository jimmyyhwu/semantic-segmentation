--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/models/init.lua
--

require 'nn'
require 'cunn'
require 'cudnn'

local datasets = require 'datasets/init'

local M = {}

function M.setup(opt, model)
   if not model then
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   local labelWeights
   if opt.medFreqBal then
      local dataset = datasets.create(opt, 'train')
      labelWeights = dataset:classWeights()
      assert(labelWeights, 'median frequency balancing enabled but class weights were not found')
   else
      -- label indices go from 2 to nClasses+1, index 1 corresponds to unlabeled pixels
      labelWeights = torch.ones(opt.nClasses+1)
      if not opt.backgroundClass then labelWeights[1] = 0 end
   end
   local criterion = cudnn.SpatialCrossEntropyCriterion(labelWeights)

   -- move to gpu
   model:cuda()
   criterion:cuda()

   -- use optnet to reduce memory usage
   if opt.optnet then
      local imsize = opt.dataset == 'ade20k_384x384' and 384 or opt.dataset == 'coco' and 384
      if imsize then
         local optnet = require 'optnet'
         local sampleInput = torch.zeros(opt.batchSize, 3, imsize, imsize):cuda()
         optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
      end
   end

   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   return model, criterion
end

return M
