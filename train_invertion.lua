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
require 'sys'
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
        learningRate = opt.LR_gan_netG,
        beta1 = opt.beta_gan_netG,
   }
   self.optimState_netD = optimState.optimState_netD or {
        learningRate = opt.LR_gan_netD,
        beta1 = opt.beta_gan_netD,
   }
   self.opt = opt
   self.params_netG, self.gradParams_netG = self.netG:getParameters()
   if self.netD then
        self.params_netD, self.gradParams_netD = self.netD:getParameters()
   end
   if opt.display then self.disp = require 'display' end
end

function TrainerInvertion:train(epoch, dataloader)
   -- Trains the model for a single epoch

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local trainSize = dataloader:size()
   local lossSum = 0.0
   local loss_l2, loss_netD, loss_netG = 0.0, 0.0, 0.0
   local loss_netC = 0.0
   local N = 0


   local function feval()
      return self.criterion.crit_mse.output, self.gradParams_netG
   end

   local fDx = function(x)
     self.netD:zeroGradParameters()
     -- real image forward/backward
     local output = self.netD:forward(self.input) -- self.input is the real image
     self.target:fill(1) -- real label
     local errD_real = self.criterion.crit_gan:forward(output, self.target)
     local df_do = self.criterion.crit_gan:backward(output, self.target)
     self.netD:backward(self.input, df_do)

     -- fake image forward/backward
     local output = self.netD:forward(self.images_gen)
     self.target:fill(0) -- fake label
     local errD_fake = self.criterion.crit_gan:forward(output, self.target)
     local df_do = self.criterion.crit_gan:backward(output, self.target)
     self.netD:backward(self.input, df_do)
     loss_netD = errD_real + errD_fake
     return loss_netD, self.gradParams_netD 
   end

   local fGx = function(x)
     self.netG:zeroGradParameters()
     self.target:fill(1)
     local output = self.netD.output
     
     loss_netG = self.criterion.crit_gan:forward(output, self.target)
     local df_do = self.criterion.crit_gan:backward(output, self.target)
     local df_dg = self.netD:updateGradInput(self.images_gen, df_do)
     self.netG:backward(self.feats, df_dg)
     return loss_netG, self.gradParams_netG
   end

   local fGx_mse = function(x)
     self.netG:zeroGradParameters()

     -- GAN loss
     self.target:fill(1)
     local grad_out = nil

     local output = self.netD.output
     loss_netG = self.criterion.crit_gan:forward(output, self.target)
     local df_do = self.criterion.crit_gan:backward(output, self.target)
     local grad_out_gan = self.netD:updateGradInput(self.images_gen, df_do)
     
     if grad_out then
         grad_out:add(self.opt.weight_loss_gan, grad_out_gan)
     else
         grad_out_gan:mul(self.opt.weight_loss_gan)
         grad_out = grad_out_gan
     end
    
     -- Pixel MSE loss
     loss_l2 = self.criterion.crit_mse:forward(self.images_gen, self.input)
     local grad_out_pixel = self.criterion.crit_mse:backward(self.images_gen, self.input)
     if grad_out then
         grad_out:add(self.opt.weight_loss_l2, grad_out_pixel)
     else
         grad_out_pixel:mul(self.opt.weight_loss_l2)
         grad_out = grad_out_pixel
     end

     self.netG:backward(self.feats, grad_out)

     return loss_netG, self.gradParams_netG
   end

   local fGx_mse_comparator = function(x)
     self.netG:zeroGradParameters()

     -- GAN loss
     self.target:fill(1)
     local grad_out = nil

     local output = self.netD.output
     loss_netG = self.criterion.crit_gan:forward(output, self.target)
     local df_do = self.criterion.crit_gan:backward(output, self.target)
     local grad_out_gan = self.netD:updateGradInput(self.images_gen, df_do)
     
     if grad_out then
         grad_out:add(self.opt.weight_loss_gan, grad_out_gan)
     else
         grad_out_gan:mul(self.opt.weight_loss_gan)
         grad_out = grad_out_gan
     end
    
     -- Pixel MSE loss
     loss_l2 = self.criterion.crit_mse:forward(self.images_gen, self.input)
     local grad_out_pixel = self.criterion.crit_mse:backward(self.images_gen, self.input)
     if grad_out then
         grad_out:add(self.opt.weight_loss_l2, grad_out_pixel)
     else
         grad_out_pixel:mul(self.opt.weight_loss_l2)
         grad_out = grad_out_pixel
     end

     -- comparator loss
     self.netC:zeroGradParameters()
     local feats_real = self.netC:forward(self.input):clone()
     local feats_fake = self.netC:forward(self.images_gen)
     loss_netC = self.criterion.crit_comparator:forward(feats_fake, feats_real)
     self.criterion.crit_comparator:backward(feats_fake, feats_real)

     local grad_out_comparator = self.netC:backward(self.images_gen, self.criterion.crit_comparator.gradInput)
     if grad_out then
         grad_out:add(self.opt.weight_loss_comparator, grad_out_comparator)
     else
         grad_out_comparator:mul(self.opt.weight_loss_comparator)
         grad_out = grad_out_comparator
     end

     grad_out:mul(self.opt.weight_overall)
     self.netG:backward(self.feats, grad_out)
     return loss_netG, self.gradParams_netG
   end


   local fCx = function(x)
     self.netC:zeroGradParameters()
     local feats_real = self.netC:forward(self.input):clone()
     local feats_fake = self.netC:forward(self.images_gen)
     loss_netC = self.criterion.crit_comparator:forward(feats_fake, feats_real)
     self.criterion.crit_comparator:backward(feats_fake, feats_real)
     self.netC:backward(self.images_gen, self.criterion.crit_comparator.gradInput)
     self.netG:zeroGradParameters()
     self.netG:backward(self.feats, self.netC.gradInput)
     return loss_netC, self.gradParams_netG
   end

   local feval_mse_comparator = function(x)
     local grad_out = nil
     self.netG:zeroGradParameters()
       
     -- l2 loss
     loss_l2 = self.criterion.crit_mse:forward(self.images_gen, self.input)
     local grad_out_pixel = self.criterion.crit_mse:backward(self.images_gen, self.input)
     if grad_out then
         grad_out:add(self.opt.weight_loss_l2, grad_out_pixel)
     else
         grad_out_pixel:mul(self.opt.weight_loss_l2)
         grad_out = grad_out_pixel
     end

     -- comparator loss
     self.netC:zeroGradParameters()
     local feats_real = self.netC:forward(self.input):clone()
     local feats_fake = self.netC:forward(self.images_gen)
     loss_netC = self.criterion.crit_comparator:forward(feats_fake, feats_real)
     self.criterion.crit_comparator:backward(feats_fake, feats_real)

     local grad_out_comparator = self.netC:backward(self.images_gen, self.criterion.crit_comparator.gradInput)
     if grad_out then
         grad_out:add(self.opt.weight_loss_comparator, grad_out_comparator)
     else
         grad_out_comparator:mul(self.opt.weight_loss_comparator)
         grad_out = grad_out_comparator
     end

     self.netG:backward(self.feats, grad_out)
     return loss_l2, self.gradParams_netG 
   end

   

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
      if self.opt.loss_l2 == 1 and self.opt.loss_gan == 0 and self.opt.loss_comparator == 0 then
        loss_l2 = self.criterion.crit_mse:forward(self.images_gen, self.input)
        self.netG:zeroGradParameters()
        self.criterion.crit_mse:backward(self.images_gen, self.input)
        self.netG:backward(self.feats, self.criterion.crit_mse.gradInput)
        optim.adam(feval, self.params_netG, self.optimState_netG)
      end

      -- Adversarial Training loss
      if self.opt.loss_l2 == 0 and self.opt.loss_gan == 1 and self.opt.loss_comparator == 0 then 
        optim.adam(fDx, self.params_netD, self.optimState_netD)
        optim.adam(fGx, self.params_netG, self.optimState_netG)
      end
      
      -- Feature loss
      if self.opt.loss_comparator == 1 and self.opt.loss_l2 == 0 and self.opt.loss_gan == 0 then
         optim.adam(fCx, self.params_netG, self.optimState_netG)
      end

      -- Joint loss for MSE and GAN
      if self.opt.loss_l2 == 1 and self.opt.loss_gan == 1 and self.opt.loss_comparator == 0 then
         optim.adam(fDx, self.params_netD, self.optimState_netD)
         optim.adam(fGx_mse, self.params_netG, self.optimState_netG) -- joint loss (GAN + MSE)
      end
      
      -- Joint loss for MSE and Feature
      if self.opt.loss_l2 == 1 and self.opt.loss_gan == 0 and self.opt.loss_comparator == 1 then
        optim.adam(feval_mse_comparator, self.params_netG, self.optimState_netG)
      end
      
      -- Joint loss for MSE and Feature and GAN
      if self.opt.loss_l2 == 1 and self.opt.loss_gan == 1 and self.opt.loss_comparator == 1 then
        optim.adam(fDx, self.params_netD, self.optimState_netD)
        optim.adam(fGx_mse_comparator, self.params_netG, self.optimState_netG)
      end

      lossSum = lossSum + loss_l2*batchSize
      N = N + batchSize

      -- visualization
      if nBatch_train % 40 == 0 and self.opt.display==1 then
        self.images_gen:add(1):mul(0.5)
        self.input:add(1):mul(0.5)
        self.disp.image(self.images_gen,{win = self.opt.displayID, title='synthesized_' .. self.opt.exp_name})
        self.disp.image(self.input,{win = self.opt.displayID*10, title='real_' .. self.opt.exp_name})
      end
 
      print(('%s | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err_l2=%1.4f Err_netD=%1.4f Err_netG=%1.4f Err_netC=%1.4f'):format(
	self.opt.exp_name, epoch, n, trainSize, timer:time().real, dataTime, loss_l2, loss_netD, loss_netG, loss_netC))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params_netG:storage() == self.netG:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return lossSum / N
end

function TrainerInvertion:test(epoch, dataloader)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local N = 0
   local lossSum = 0.0
   self.netG:evaluate()
   local nBatch_test = 0
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      nBatch_test = nBatch_test + 1
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

      sys.sleep(2)
         -- visualization
      --if nBatch_test % 40 == 0 and self.opt.display==1 then
        self.images_gen:add(1):mul(0.5)
        self.input:add(1):mul(0.5)
        self.disp.image(self.images_gen,{win = self.opt.displayID, title='TESTING synthesized ' .. self.opt.modelFile})
        self.disp.image(self.input,{win = self.opt.displayID*10, title='TESTING real ' .. self.opt.modelFile})
      --end
 
      timer:reset()
      dataTimer:reset()
   end

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
   -- rescale the images (very important)
   self.input:mul(2):add(-1)

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
