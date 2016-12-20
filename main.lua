--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/main.lua
--
-- Train or test semantic segmentation model
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'

local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- resume from checkpoint if exists
local checkpoint, model, optimState = checkpoints.latest(opt)
local model, criterion = models.setup(opt, model)
local DataLoader = opt.nThreads == 1 and require 'dataloader' or require 'dataloader2'
local trainLoader, valLoader = DataLoader.create(opt)
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local _, pixelAccuracy = trainer:test(0, valLoader, {})
   print(string.format(' * Results    pixel accuracy %.3f', pixelAccuracy))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or 1
local bestPixelAccuracy = checkpoint and checkpoint.bestPixelAccuracy or 0
local trainLog = checkpoint and checkpoint.trainLog or {}
local testLog = checkpoint and checkpoint.testLog or {}
for epoch = startEpoch, opt.nEpochs do
   trainer:train(epoch, trainLoader, trainLog)

   -- test on validation set after every train epoch
   local _, testPixelAccuracy = trainer:test(epoch, valLoader, testLog)

   local isBestModel = false
   if testPixelAccuracy > bestPixelAccuracy then
      isBestModel = true
      bestPixelAccuracy = testPixelAccuracy
      print(string.format(' * Best model    pixel accuracy %.3f', testPixelAccuracy))
   end

   checkpoint = {
      epoch = epoch,
      trainLog = trainLog,
      testLog = testLog,
      bestPixelAccuracy = bestPixelAccuracy,
      isBestModel = isBestModel,
   }
   checkpoints.save(trainer.lightModel, trainer.optimState, checkpoint, opt)
   print('')
end

print(string.format(' * Finished    pixel accuracy %.3f', bestPixelAccuracy))
