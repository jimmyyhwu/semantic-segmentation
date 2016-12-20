--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/checkpoints.lua
--
-- Checkpoint loading and saving
--

local M = {}

-- Sanitize gradients to reduce checkpoint size.
-- See https://github.com/karpathy/neuraltalk2/blob/master/misc/net_utils.lua
local function sanitizeGradients(model)
   for _, m in ipairs(model:listModules()) do
      if m.weight and m.gradWeight then m.gradWeight = nil end
      if m.bias and m.gradBias then m.gradBias = nil end
   end
end

local function unsanitizeGradients(model)
   for _, m in ipairs(model:listModules()) do
      if m.weight and not m.gradWeight then
         m.gradWeight = m.weight:clone():zero()
      end
      if m.bias and not m.gradBias then
         m.gradBias = m.bias:clone():zero()
      end
   end
end

function M.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)

   local modelPath = paths.concat(opt.resume, latest.modelFile)
   assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
   print('=> Resuming model from ' .. modelPath)
   local model = torch.load(modelPath):cuda()
   unsanitizeGradients(model)

   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, model, optimState
end

function M.save(lightModel, optimState, checkpoint, opt)
   sanitizeGradients(lightModel)

   -- save model
   local modelFile = 'model_' .. checkpoint.epoch .. '.t7'
   torch.save(paths.concat(opt.save, modelFile), lightModel)

   -- save optim state
   local optimFile = 'optimState_' .. checkpoint.epoch .. '.t7'
   torch.save(paths.concat(opt.save, optimFile), optimState)

   -- save rest of checkpoint data
   checkpoint['modelFile'] = modelFile
   checkpoint['optimFile'] = optimFile
   torch.save(paths.concat(opt.save, 'latest.t7'), checkpoint) 

   if checkpoint.isBestModel then
      torch.save(paths.concat(opt.save, 'model_best.t7'), lightModel)
   end
end

return M
