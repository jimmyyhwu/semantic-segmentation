--
-- Adapted from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/README.md
--
-- Setup code for semantic segmentation on the COCO dataset
--

local coco = require 'coco'
local ffi = require 'ffi'
local image = require 'image'

local M = {}

local imageSize = 384

-- convert list of object instance annotations into a single
-- semantic segmentation annotation mask
local function annsToAnn(anns, h, w, catIdToClassIdx)
   -- class indices in the annotations range from 1 to nClasses+1,
   -- where index 1 corresponds to unlabeled pixels
   local ann = torch.ByteTensor(h, w):fill(1)
   local rle
   for i = 1, #anns do
      -- see section 4.1 at http://mscoco.org/dataset/#download
      -- for explanation of annotation fields
      if (anns[i].iscrowd == 0) then
         rle = coco.MaskApi.frPoly(anns[i].segmentation, h, w)
      else
         rle = anns[i].segmentation
      end
      local mask = coco.MaskApi.decode(rle)
      local classIdx = catIdToClassIdx[anns[i].category_id]
      ann[mask:eq(1)] = classIdx -- undefined behavior if masks overlap
   end
   return ann
end

local function processSplit(datadir, gendir, split, catIdToClassIdx, nClasses)
   local annPath = paths.concat(datadir, 'annotations/instances_' .. split .. '2014.json')
   assert(paths.filep(annPath), 'annotation file not found: ' .. annPath)
   local cocoApi = coco.CocoApi(annPath)

   -- save resized images and annotation masks
   local imgDir = paths.concat(gendir, 'coco', split .. '2014')
   if not paths.filep(imgDir) then paths.mkdir(imgDir) end

   local imgPaths = {}
   local annPaths = {}
   local maxLenImgPath = -1
   local maxLenAnnPath = -1
   local rgbSum = torch.zeros(3)
   local nPixelsSum = 0

   -- calculate class weights for median frequency balancing
   local counts, presence
   if nClasses then
      counts = torch.zeros(nClasses+1)
      presence = torch.zeros(nClasses+1)
   end

   local imgIds = cocoApi:getImgIds()
   for i = 1, imgIds:numel() do
      local imgId = imgIds[i]
      local imgInfo = cocoApi:loadImgs(imgId)[1]

      -- resize image and save
      local imgPathOrig = paths.concat(datadir, split .. '2014', imgInfo.file_name)
      local img = image.load(imgPathOrig, 3, 'byte'):float()
      img = image.scale(img, imageSize, imageSize)
      local nPixels = img:size(2)*img:size(3)
      rgbSum = rgbSum + img:reshape(3, nPixels):sum(2)
      nPixelsSum = nPixelsSum + nPixels
      local imgPath = paths.concat(imgDir, imgInfo.file_name)
      maxLenImgPath = math.max(maxLenImgPath, #imgPath + 1)
      image.save(imgPath, img:byte()) -- must save using byte format
      table.insert(imgPaths, imgPath)

      -- create annotation mask with matching size and save
      local anns = cocoApi:loadAnns(cocoApi:getAnnIds({imgId=imgId}))
      local ann = annsToAnn(anns, imgInfo.height, imgInfo.width, catIdToClassIdx)
      ann = image.scale(ann, imageSize, imageSize, 'simple') -- no interpolation

      -- update counts for frequency calculation
      if nClasses then
         local c = torch.histc(ann:float(), nClasses+1, 1, nClasses+1)
         counts:add(c)
         local p = torch.zeros(nClasses+1)
         p[c:gt(0)] = nPixels
         presence:add(p)
      end

      local annPath = paths.concat(imgDir, paths.basename(imgInfo.file_name, '.jpg') .. '.png')
      maxLenAnnPath = math.max(maxLenAnnPath, #annPath + 1)
      image.save(annPath, ann)
      table.insert(annPaths, annPath)

      xlua.progress(i, imgIds:numel())
   end

   -- compute class weights
   local freqs
   if nClasses then
      freqs = torch.cdiv(counts, presence)
      local median = torch.median(freqs):squeeze()
      freqs:cinv()
      freqs:mul(median)
      freqs[freqs:ne(freqs)] = 0 -- remove nans
   end

   -- convert path lists to tensors for faster loading
   local imgPath = torch.CharTensor(#imgPaths, maxLenImgPath):zero()
   for i, p in ipairs(imgPaths) do ffi.copy(imgPath[i]:data(), p) end
   local annPath = torch.CharTensor(#annPaths, maxLenAnnPath):zero()
   for i, p in ipairs(annPaths) do ffi.copy(annPath[i]:data(), p) end

   return imgPath, annPath, rgbSum / nPixelsSum, freqs
end

-- read categories from annotation file and define class indices
local function findClasses(dir)
   local catIdToClassIdx, classList
   for _, split in ipairs{'train', 'val'} do
      local annPath = paths.concat(dir, 'annotations/instances_' .. split .. '2014.json')
      local cocoApi = coco.CocoApi(annPath)
      local catIds = cocoApi:getCatIds()
      if catIdToClassIdx then
         assert(catIds:size(1) == #classList, 'train and val must contain same annotation classes')
      else
         catIdToClassIdx = {}
         classList = {}
         for i = 1, catIds:size(1) do
            -- these indices range from 2 to nClasses+1, 1 is reserved for unlabeled
            catIdToClassIdx[catIds[i]] = i+1
            -- these indices range from 1 to nClasses
            classList[i] = cocoApi:loadCats(catIds[i])[1].name
         end
      end
   end
   return catIdToClassIdx, classList
end

function M.exec(opt, cacheFile)
   print('=> Setting up COCO for semantic segmentation')
   local catIdToClassIdx, classList = findClasses(opt.data)

   print(' | processing train images and annotations')
   local trainImgPath, trainAnnPath, rgbMean, classWeights = processSplit(
      opt.data, opt.gen, 'train', catIdToClassIdx, opt.nClasses)

   print(' | processing val images and annotations')
   local valImgPath, valAnnPath = processSplit(opt.data, opt.gen, 'val', catIdToClassIdx)

   local info = {
      basedir = opt.data,
      classList = classList,
      imageSize = imageSize,
      mean = rgbMean,
      classWeights = classWeights,
      train = {
         imgPath = trainImgPath,
         annPath = trainAnnPath,
      },
      val = {
         imgPath = valImgPath,
         annPath = valAnnPath,
      },
   }

   print(' | saving image info to ' .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
