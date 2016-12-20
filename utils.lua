local cjson = require 'cjson'
local gnuplot = require 'gnuplot'

local M = {}

-- plot semantic segmentation annotation mask
function M.plotAnnotation(annotation, path, colormap)
   function colorize(annotation)
      local annotationColorized = annotation.new()
      annotation.image.colorize(annotationColorized, annotation-1, colormap)
      return annotationColorized
   end

   image.save(path, colorize(annotation))
end

-- plot loss/accuracy as a function of iteration/epoch
function M.plotLog(tbl, key, xlabel, path, logscale)
   local plots = {}
   table.insert(plots, {key, torch.Tensor(tbl[key]), '-'})
   local fig = gnuplot.pngfigure(path)
   if logscale then gnuplot.logscale('on') end
   gnuplot.plot(plots)
   -- gnuplot.title('')
   gnuplot.grid('on')
   gnuplot.xlabel(xlabel)
   gnuplot.plotflush()
   gnuplot.close(fig)
end

function M.readJson(path)
   local f = io.open(path, 'r')
   local tbl = cjson.decode(f:read('*all'))
   f:close()
   return tbl
end

function M.writeJson(tbl, path)
   local f = io.open(path, 'w')
   f:write(cjson.encode(tbl))
   f:close()
end

return M
