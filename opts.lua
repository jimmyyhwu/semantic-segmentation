--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/opts.lua
--

local M = {}

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Semantic Segmentation Training Script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options ---------------------
   cmd:option('-dataset', 'ade20k_384x384', 'name of dataset: ade20k_384x384 | coco')
   cmd:option('-data', 'data/ade20k_384x384', 'dataset directory')
   cmd:option('-manualSeed', 0, 'rng seed')
   cmd:option('-backend', 'cudnn', 'options: cudnn')
   cmd:option('-cudnn', 'fastest', 'options: fastest | default | deterministic')
   cmd:option('-gen', 'gen', 'directory to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads', 2, 'number of data loader threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs', 0, 'number of epochs to train for')
   cmd:option('-batchSize', 4, 'mini-batch size')
   cmd:option('-testOnly', 'false', 'run on validation set only')
   cmd:option('-plotOutput', 'true', 'save output segmentations for validation set')
   cmd:option('-log', 'true', 'write train and val losses to json during training')
   ------------- Checkpointing options ---------------
   cmd:option('-save', 'checkpoints', 'directory to save checkpoints')
   cmd:option('-resume', 'none', 'checkpoint directory to resume from')
   ------------- Optimization options ----------------
   cmd:option('-optimMethod', 'sgd', 'options: sgd | adam')
   cmd:option('-LR', 1e-4, 'initial learning rate')
   cmd:option('-LRDecayEpochs', 20, 'drop learning rate by 0.1 every LRDecayEpochs epochs')
   cmd:option('-momentum', 0.9, 'momentum for sgd')
   cmd:option('-beta1', 0.9, 'first moment coefficient for adam')
   cmd:option('-beta2', 0.999, 'second moment coefficient for adam')
   cmd:option('-weightDecay', 5e-4, 'weight decay')
   ------------- Model options -----------------------
   cmd:option('-netType', 'dilated', 'options: dilated | segnet')
   cmd:option('-prototxt', 'caffemodel/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to vgg-16 caffe prototxt')
   cmd:option('-caffemodel', 'caffemodel/VGG_ILSVRC_16_layers.caffemodel', 'path to vgg-16 caffemodel')
   cmd:option('-optnet', 'true', 'use optnet to reduce memory usage')
   cmd:option('-nClasses', 0, 'number of instance classes in dataset')
   cmd:option('-backgroundClass', 'false', 'treat unlabeled pixels as a class of its own')
   cmd:option('-medFreqBal', 'false', 'use median frequency balancing')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.plotOutput = opt.plotOutput ~= 'false'
   opt.backgroundClass = opt.backgroundClass ~= 'false'
   opt.medFreqBal = opt.medFreqBal ~= 'false'

   if opt.optimMethod ~= 'sgd' and opt.optimMethod ~= 'adam' then
      cmd:error('unknown optim method: ' .. opt.optimMethod)
   end

   if opt.dataset == 'ade20k_384x384' and opt.nThreads > 1 then
      opt.nThreads = 1 -- hdf5 doesn't work with multi-threaded
   end

   if opt.medFreqBal and not opt.backgroundClass then
      cmd:error('median frequency balancing must include background class')
   end

   if opt.dataset ~= 'coco' and opt.medFreqBal then
      cmd:error('median frequency balancing not supported for dataset: ' .. opt.dataset)
   end

   if opt.dataset == 'ade20k_384x384' then
      opt.nEpochs = opt.nEpochs == 0 and 100 or opt.nEpochs
      opt.nClasses = opt.nClasses == 0 and 150 or opt.nClasses
   elseif opt.dataset == 'coco' then
      opt.nEpochs = opt.nEpochs == 0 and 100 or opt.nEpochs
      opt.nClasses = opt.nClasses == 0 and 80 or opt.nClasses
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if not paths.dirp(opt.save) then paths.mkdir(opt.save) end
   if not paths.dirp(opt.gen) then paths.mkdir(opt.gen) end

   -- enable logging if training
   if not opt.testOnly then
      local logPath = paths.concat(opt.save, 'log')
      local logFile = (opt.resume ~= 'none' and paths.filep(logPath)) and io.open(logPath, 'a') or logPath
      cmd:log(logFile, opt)
      cmd:addTime(opt.save, '%F %T')
   end

   return opt
end

return M
