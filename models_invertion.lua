require 'nn'
require 'cunn'
require 'cudnn'
require 'utils'

local M = {}

local function build_netD(opt)
-- netD: discriminator
	local SpatialBatchNormalization = nn.SpatialBatchNormalization
	local SpatialConvolution = nn.SpatialConvolution
	local SpatialFullConvolution = nn.SpatialFullConvolution
	
    local netD = nn.Sequential()
    local ndf = opt.ndf
    local nc = opt.nc
    -- input is (nc) x 256x256
    netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 64 x 64
    netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 32 x 32
    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 16 x 16
    netD:add(SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 8 x 8
    --netD:add(nn.SpatialAveragePooling(8,8,1,1)) 
    --netD:add(nn.View(ndf*8):setNumInputDims(3))
    --netD:add(nn.Linear(ndf*8, 2))
    netD:add(nn.SpatialConvolution(ndf*8, 1, 8, 8))
    netD:add(nn.Sigmoid())
    netD:add(nn.View(1):setNumInputDims(3))
    
    netD:apply(weights_init)
    return netD
end

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
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.3, true))
	    -- state size: (ngf*8) x 4 x 4
	    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.3, true))
	    -- state size: (ngf*4) x 8 x 8
	    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.3, true))
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
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2,true))
	    -- state size: (ngf*8) x 4 x 4
	    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2,true))
	    -- state size: (ngf*4) x 8 x 8
	    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2,true))
	      -- state size: (ngf*2) x 16 x 16
	    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2,true))
	     -- state size: (ngf) x 32 x 32
	    netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
	    netG:add(nn.Tanh())
        --netG:add(nn.Sigmoid())

    elseif type_netG == 4 then
        netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2,true))
	    -- state size: (ngf*8) x 4 x 4
	    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2,true))
	    -- state size: (ngf*4) x 8 x 8
	    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2,true))
		-- upsample one more time
        netG:add(SpatialFullConvolution(ngf * 2, ngf * 2, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2,true))
	      -- state size: (ngf*2) x 16 x 16
	    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2,true))
    	-- upsample one more time
        netG:add(SpatialFullConvolution(ngf, ngf, 4, 4, 2, 2, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2,true))
	    netG:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
	    netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2,true))
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
   if opt.model_pretrain == 'pretrained/resnet-34.t7' then
        local netF = torch.load(opt.model_pretrain)
        if opt.resnet_layer == 'conv' then
            netF:remove(#netF.modules)
            netF:remove(#netF.modules) -- output: 10x512x1x1 (if the input is 3x224x224)
            netF:remove(#netF.modules) -- output: 10x512x7x7 (if the input is 3x224x224)
            opt.exp_name = 'resnet-conv_' .. opt.exp_name
            opt.type_netG = 3
        elseif opt.resnet_layer == 'fc' then
            netF:remove(#netF.modules)
            netF:remove(#netF.modules)
            netF:remove(#netF.modules)
            netF:add(nn.SpatialAveragePooling(8,8)) -- the input is 256x256
            opt.exp_name = 'resnet-fc_' .. opt.exp_name
            opt.type_netG = 4 -- decode the 512 feature vector into the image
        end
        model.netF = netF
   --netF:add(nn.SpatialAveragePooling(8,8))                           -- output: 10x512x8x8 (if the input is 3x256x256)
   elseif opt.model_pretrain == 'pretrained/places2_alexnet.t7' then
        opt.type_netG = 4
        opt.exp_name = 'places2_alexnet_' .. opt.exp_name
        local netF = torch.load(opt.model_pretrain)
        netF:remove() -- output: (batch_size x 4096) 
        netF:remove()
        netF:remove()
        netF:remove()
        netF:remove()
        model.netF = netF
   end

   
   model.netF:cuda()
   model.netF:evaluate()

   if opt.loss_comparator == 1 then
       -- create the feature comparator loss inside 
       local netC = torch.load(opt.model_comparator) -- places2_CNN
       --for i=28,18,-1 do netC:remove(i) end -- pool5 CNN
       if opt.model_comparator == 'pretrained/places2_alexnet.t7' then 
           --for i=28, 12, -1 do netC:remove(i) end -- conv3 CNN
           for i=28, 8, -1 do netC:remove(i) end
       else
           print('no comparator')
       end
       netC:cuda()
       model.netC = netC
       print('Comparator network:')
       print(model.netC) 
   end
   
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
      local netD = build_netD(opt)
      model.netD = netD:cuda()
      model.netG = netG:cuda()
		
   end
    print('testing netF and netG...')
    local input = torch.zeros(10,3,256,256)
    input = input:cuda()
    local feats = model.netF:forward(input)
    local image_sync = model.netG:forward(feats)
    print(image_sync:size())
    print('testing passed')
    local crit = {}
    crit.crit_mse = nn.MSECriterion():cuda()
    crit.crit_comparator = nn.MSECriterion():cuda()
    crit.crit_gan = nn.BCECriterion():cuda()



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
