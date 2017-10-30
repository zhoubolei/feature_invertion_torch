require 'nn'
require 'cunn'
require 'cudnn'
require 'utils'

local M = {}


local function build_netG(opt)
-- build the image generator 
	local netG = nn.Sequential()
    local nz = opt.nz
    local ngf = opt.ngf
    local nc = opt.nc

	local SpatialBatchNormalization = nn.SpatialBatchNormalization
	local SpatialConvolution = nn.SpatialConvolution
	local SpatialFullConvolution = nn.SpatialFullConvolution
	
	local type_netG = opt.type_netG
	
	if type_netG == 1 then
	    -- the naive architecture from DCGAN (512x1x1)
	    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
	    -- state size: (ngf*8) x 4 x 4
	    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
	    -- state size: (ngf*4) x 8 x 8
	    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
	    -- state size: (ngf*2) x 16 x 16
	    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
	    -- state size: (ngf) x 32 x 32
	    netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
	    netG:add(nn.Tanh())
	
	elseif type_netG == 2 then
	    -- input should be (512x8x8)
	    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
	    -- state size: (ngf*8) x 4 x 4
	    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
	    -- state size: (ngf*4) x 8 x 8
	    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true)) -- state size: (ngf*2) x 16 x 16
	    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
	     -- state size: (ngf) x 32 x 32
	    netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
	    netG:add(nn.Tanh())
	
	elseif type_netG == 3 then
	    -- input should be (512x8x8) ( more convolution)
	    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
	    netG:add(SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
	    -- state size: (ngf*8) x 4 x 4
	    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
	    netG:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
	    -- state size: (ngf*4) x 8 x 8
	    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
	    netG:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
	      -- state size: (ngf*2) x 16 x 16
	    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
	    netG:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
	     -- state size: (ngf) x 32 x 32
	    netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
	    netG:add(nn.Tanh())
	
	end
	netG:apply(weights_init)
    return netG
end

function M.setup(opt, checkpoint)
   local model = {}
  -- load the feature extractor network (pretrained imagenet model)
   local netF = torch.load(opt.model_pretrain)
   netF:remove(#netF.modules)
   netF:remove(#netF.modules) -- output: 10x512x1x1 (if the input is 3x224x224)
   netF:remove(#netF.modules) -- output: 10x512x7x7 (if the input is 3x224x224)
   netF:cuda()
   netF:evaluate()
   model.netF = netF


   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      local nets = torch.load(modelPath)
      model.netD = nets.netD:cuda()
      model.netG = nets.netG:cuda()

   else
      print('=> Creating new generator network and discriminator network')
      local netG = build_netG(opt)
      model.netG = netG:cuda()
		
   end


    local crit = {}
    crit.crit_mse = nn.MSECriterion():cuda()



   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   return model, crit
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

return M
