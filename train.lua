--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/train.lua
--
-- Training and testing code
--

local optim = require 'optim'
local utils = require 'utils'

local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.opt = opt

   -- initialize optim state
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      weightDecay = opt.weightDecay,
   }
   if opt.optimMethod == 'sgd' then
      self.optimState['momentum'] = opt.momentum
      self.optimState['dampening'] = 0.0
      self.optim = optim.sgd
   elseif opt.optimMethod == 'adam' then
      self.optimState['beta1'] = opt.beta1
      self.optimState['beta2'] = opt.beta2
      self.optim = optim.adam
   end

   -- log train and test losses to json
   if self.opt.log and not self.opt.testOnly then
      self.trainLogFile = paths.concat(opt.save, 'train_log.json')
      self.testLogFile = paths.concat(opt.save, 'test_log.json')
   end

   self.params, self.gradParams = model:getParameters()

   model:clearState()
   self.lightModel = model:clone('weight', 'bias', 'running_mean', 'running_var')
end

function Trainer:train(epoch, dataloader, trainLog)
   -- learning rate step decay
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   if not trainLog.loss then trainLog['loss'] = {} end

   local trainSize = dataloader:size()
   local lossSum = 0.0
   local N = 0

   print('=> Training epoch ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real -- includes dataloader collectgarbage time

      -- copy to gpu
      self:copySample(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      -- record training loss every iteration
      table.insert(trainLog.loss, loss)
      if self.opt.log then
         utils.writeJson(trainLog, self.trainLogFile)
         utils.plotLog(trainLog, 'loss', 'iterations', paths.concat(self.opt.save, 'train_loss.png'))
      end

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      self.optim(feval, self.params, self.optimState)

      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      print((' | Train: [%d][%d/%d]    time %.3fs  data %.3fs  loss %.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss))

      -- check that model:parameters was not called again
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return lossSum / N
end

function Trainer:test(epoch, dataloader, testLog)
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local pixelsCorrectSum, pixelsLabeledSum = 0.0, 0.0
   local lossSum = 0.0
   local N = 0

   -- save output segmentations to png files
   if self.opt.plotOutput then
      self.outputDir = paths.concat(self.opt.save, 'output_' .. epoch)
      if not paths.dirp(self.outputDir) then paths.mkdir(self.outputDir) end
      if not self.colormap then
         self.colormap=torch.zeros(self.opt.nClasses+1, 3)
         self.colormap[{ {2,self.opt.nClasses+1}, {} }] = image.colormap(self.opt.nClasses)
      end
   end

   if not testLog.loss then testLog['loss'] = {} end
   if not testLog['pixel accuracy'] then testLog['pixel accuracy'] = {} end

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      self:copySample(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      -- convert probabilities to segmentation mask
      output = select(2, output:max(2)):squeeze(2)
      if self.opt.plotOutput then
         -- save output segmentation mask
         self:plotOutput(epoch, n, batchSize, sample.target, output:float())
      end

      local pixelsCorrect, pixelsLabeled = self:computePixelAccuracy(output:byte(), sample.target:byte())
      pixelsCorrectSum = pixelsCorrectSum + pixelsCorrect
      pixelsLabeledSum = pixelsLabeledSum + pixelsLabeled

      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    time %.3fs  data %.3fs  loss %.3f  pixel accuracy %.3f'):format(
         epoch, n, size, timer:time().real, dataTime, loss, pixelsCorrect/pixelsLabeled))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   local pixelAccuracy = pixelsCorrectSum / pixelsLabeledSum

   print((' * Finished epoch %d    loss %.3f  pixel accuracy %.3f'):format(
      epoch, lossSum / N, pixelAccuracy))

   -- record loss and pixel accuracy on validation set for every epoch
   table.insert(testLog.loss, lossSum / N)
   table.insert(testLog['pixel accuracy'], pixelAccuracy)

   -- log test loss and pixel accuracy to json
   if self.opt.log and not self.opt.testOnly then
      utils.writeJson(testLog, self.testLogFile)
      utils.plotLog(testLog, 'loss', 'epochs', paths.concat(self.opt.save, 'test_loss.png'))
      utils.plotLog(testLog, 'pixel accuracy', 'epochs', paths.concat(self.opt.save, 'test_pixel_accuracy.png'))
   end

   return lossSum / N, pixelAccuracy
end

function Trainer:computePixelAccuracy(output, target)
   local labeled = self.opt.backgroundClass and 0 or 1
   local pixelsCorrect = output:eq(target):cmul(target:gt(labeled)):sum()
   local pixelsLabeled = target:gt(labeled):sum()
   return pixelsCorrect, pixelsLabeled
end

function Trainer:plotOutput(epoch, batchNum, batchSize, target, output)
   function getIndex(batchSize, batchNum, i)
      return batchSize*(batchNum-1) + i
   end

   for i = 1, batchSize do
      local iter = getIndex(batchSize, batchNum, i)
      utils.plotAnnotation(target[i], paths.concat(self.outputDir, iter .. '_target.png'), self.colormap)
      utils.plotAnnotation(output[i], paths.concat(self.outputDir, iter .. '_output.png'), self.colormap)
   end
end

-- copy sample to gpu
function Trainer:copySample(sample)
   self.input = self.input or torch.CudaTensor()
   self.target = self.target or torch.CudaTensor()
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

-- drop learning rate by 0.1 every LRDecayEpochs epochs
function Trainer:learningRate(epoch)
   local decay = math.floor((epoch - 1) / self.opt.LRDecayEpochs)
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
