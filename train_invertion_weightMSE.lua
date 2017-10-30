--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local TrainerInvertion = torch.class('resnet.TrainerInvertion', M)

function TrainerInvertion:__init(net, criterion, opt, optimState)
   self.netF = net.netF
   self.netG = net.netG
   self.netD = net.netD
   if opt.loss_comparator==1 then
      self.netC = net.netC
   end
   self.criterion = criterion
   self.optimState_netG = optimState.optimState_netG or {
        learningRate = opt.LR_gan,
        beta1 = opt.beta1_gan,
   }
   self.opt = opt
   self.params_netG, self.gradParams_netG = self.netG:getParameters()
   if opt.display then self.disp = require 'display' end
end

function TrainerInvertion:train(epoch, dataloader)
   -- Trains the model for a single epoch

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local trainSize = dataloader:size()
   local lossSum = 0.0
   local loss_l2 = 0.0
   local N = 0

   local function feval()
      return self.criterion.crit_mse.output, self.gradParams_netG
   end

   -- contruct the High-Low Frequency Kernel
  -- local filter_kernel = 1.0 - torch.Tensor({{1,2,1},{2,4,2},{1,2,1}}):mul(self.opt.factor/16.0) 
   local size_kernel = 9
   local gaussian_filter = image.gaussian(size_kernel)
   gaussian_filter:div(gaussian_filter:sum())
   local filter_kernel = 1.0 - gaussian_filter:mul(self.opt.factor)
   local filter_layer = nn.SpatialConvolution(3, 3, size_kernel, size_kernel, 1, 1, 4, 4)
   local para_filter = filter_layer:parameters()
   para_filter[2]:zero() -- set the bias term as 0
   para_filter[1]:zero()
   for i = 1, 3 do
        para_filter[1][i][i]:copy(filter_kernel)
   end
   filter_layer:cuda()
   local filter_layer_real = filter_layer:clone('weight','bias')
   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.netG:training()
   local nBatch_train = 0
   for n, sample in dataloader:run() do
      nBatch_train = nBatch_train + 1
      local dataTime = dataTimer:time().real
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      
      self.feats = self.netF:forward(self.input)
      self.images_gen = self.netG:forward(self.feats)
     
      local batchSize = self.feats:size(1)

      -- L2 loss (image space loss)
      if self.opt.loss_l2 == 1 then
        loss_l2 = self.criterion.crit_mse:forward(self.images_gen, self.input)
        self.netG:zeroGradParameters()
        self.criterion.crit_mse:backward(filter_layer:forward(self.images_gen), filter_layer_real:forward(self.input))
        local filter_gradInput = filter_layer:backward(self.images_gen, self.criterion.crit_mse.gradInput)
        self.netG:backward(self.feats, filter_gradInput)
        optim.adam(feval, self.params_netG, self.optimState_netG)
      end


      lossSum = lossSum + loss_l2*batchSize
      N = N + batchSize

      -- visualization
      if nBatch_train % 40 == 0 and self.opt.display==1 then
        self.disp.image(self.images_gen,{win = self.opt.displayID, title='synthesized_' .. self.opt.exp_name .. '_factor=' .. self.opt.factor})
        self.disp.image(self.input,{win = self.opt.displayID*10, title='real_' .. self.opt.exp_name .. '_factor=' .. self.opt.factor})
      end
 
      print(('%s | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err_l2=%1.4f factor=%.2f'):format(self.opt.exp_name, epoch, n, trainSize, timer:time().real, dataTime, loss_l2, self.opt.factor))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params_netG:storage() == self.netG:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return lossSum / N
end

function TrainerInvertion:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local N = 0
   local lossSum = 0.0
   self.netG:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      self.feats = self.netF:forward(self.input)
      self.images_gen = self.netG:forward(self.feats)
      local batchSize = self.feats:size(1)
     
      -- L2 loss (image space loss)
      local loss_l2 = self.criterion.crit_mse:forward(self.images_gen, self.input)

      lossSum = lossSum + loss_l2*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f'):format(
         epoch, n, size, timer:time().real, dataTime, loss_l2))

      timer:reset()
      dataTimer:reset()
   end
   --self.model:training()

--   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
--      epoch, top1Sum / N, top5Sum / N))

   return lossSum / N
end

function TrainerInvertion:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function TrainerInvertion:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function TrainerInvertion:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.TrainerInvertion
