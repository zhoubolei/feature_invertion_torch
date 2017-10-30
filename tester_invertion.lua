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
local TesterInvertion = torch.class('resnet.TesterInvertion', M)
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)


function TesterInvertion:__init(net, criterion, opt, optimState)
   self.netF = net.netF
   self.netG = net.netG
   self.netD = net.netD
   if opt.loss_comparator==1 then
      self.netC = net.netC
   end
   self.criterion = criterion
   self.optimState_netG = optimState or optimState.optimState_netG or {
        learningRate = opt.LR_gan_netG,
        beta1 = opt.beta_gan_netG,
   }
   self.optimState_netD = optimState or optimState.optimState_netD or {
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

function TesterInvertion:invert_batch(epoch, dataloader)
-- Invert the feature map of Resnet34
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

function TesterInvertion:getStat_code(dataloader)
-- Get the statistics of the codes: maximum value

   local N = 0
   local lossSum = 0.0
   self.netG:evaluate()
   local nBatch_test = 0
   local unit_max = torch.zeros(512) -- the maximum value for some unit
   unit_max = unit_max:cuda()
   local getMax_layer = nn.Sequential()
   getMax_layer:add(nn.SpatialMaxPooling(8,8))
   getMax_layer:add(nn.Squeeze())
   getMax_layer:cuda()
   for n, sample in dataloader:run() do
      nBatch_test = nBatch_test + 1
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      self.feats = self.netF:forward(self.input)
      local batchSize = self.feats:size(1)
       
      local codes = self.feats:clone()
      local codes_max = getMax_layer:forward(codes)
      local unit_max_current = torch.max(codes_max, 1)[1]
      unit_max:cmax(unit_max_current)
      require('fb.debugger').enter()
      print((' | Test: [%d/%d]   '):format(n, size))
 
   end
   unit_max = unit_max:float()
   torch.save('pretrained/unit512_max_resnet-34.t7', unit_max)
   return lossSum / N
end

function TesterInvertion:guided_generation(netT, dataloader)
-- to check how important the initialization code is to generate a real image

    local size_featuremap = 7
    local lr_code = 0.05
    -- the code upperbound and lowerbound (clipping the code)
    local unit_max = torch.load('pretrained/unit512_max_resnet-34.t7') -- estimated from the validation set
    unit_max = unit_max:view(1,512,1,1)
    local codes_upperbound = torch.expand(unit_max, self.opt.batchSize, 512, size_featuremap, size_featuremap)
    local codes_lowerbound = torch.zeros(self.opt.batchSize, 512, size_featuremap, size_featuremap)
    codes_upperbound = codes_upperbound:cuda()
    codes_lowerbound = codes_lowerbound:cuda()
    codes_upperbound:fill(5)
    local batch_id = 0 
    for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
        batch_id = batch_id + 1
        self:copyInputs(sample)
        self.feats = self.netF:forward(self.input)
        self.input:add(1):mul(0.5)
        local codes_sample = self.feats:clone()
        local images_gen = self.netG:forward(self.feats)
        local batchSize = self.feats:size(1) 
        local one_hots = torch.zeros(batchSize, 401)
        one_hots = one_hots:cuda()
        one_hots:mul(5) -- 5 seems the best?
        local categories_select = {283, 315, 70, 110, 271, 116, 174, 173, 200, 210} 
        local img_montage_real = montage(sample.input)
        local save_name = string.format('results/category_transfer/batch%d.jpg', batch_id) 
        image.save(save_name, img_montage_real)
        images_gen:add(1):mul(0.5)
        local img_montage_inverse = montage(images_gen)
        save_name = string.format('results/category_transfer/batch%d_inverse.jpg', batch_id) 
        image.save(save_name, img_montage_inverse)
           
        for idx = 1, 401 do 
            local target_idx = idx--categories_select[idx]
            local category = netT.categories[target_idx] 
            local codes = codes_sample:clone()
            one_hots:fill(0)
            one_hots[{{},target_idx}]:fill(10) -- 5 seems the best?
      
        -- clip the codes before the start
            codes:cmax(codes_lowerbound)
            codes:cmin(codes_upperbound)
            for i = 1, 55 do
                self.netG:zeroGradParameters()
                netT:zeroGradParameters()
        
                local images_gen = self.netG:forward(codes)
                local max_image = torch.max(images_gen)
                local min_image = torch.min(images_gen)
                local pred_class = netT:forward(images_gen)
                
                local df_images_gen = netT:backward(images_gen, one_hots)
                -- clip the images? 
                local norm_df_images = df_images_gen:norm()
                --df_images_gen:mul(1/norm_df_images)
                --images_gen:add(df_images_gen:mul(lr_code))
        
                local df_codes = self.netG:backward(codes, df_images_gen)
        
                local norm_df_codes = df_codes:norm()
                --df_codes:mul(10/norm_df_codes)
                codes:add(df_codes:mul(lr_code))
                -- clip the codes during the training
                local max_code = torch.max(codes)
                local min_code = torch.min(codes)
            
       --         codes:cmax(codes_lowerbound)
       --         codes:cmin(codes_upperbound)
        
                images_gen:add(1):mul(0.5)
                self.images_gen = images_gen
                local string_title = string.format('Batch%d Change to %s || iter %d  norm_dF_image=%.3f norm_df_code=%.3f max_code=%.2f',batch_id,  netT.categories[target_idx], i, norm_df_images, norm_df_codes, max_code) 
                self.disp.image(images_gen, {win = self.opt.displayID, title=string_title})
                self.disp.image(self.input, {win = self.opt.displayID+1, title=string_title})
                print(string_title)
            end
            local img_montage = montage(self.images_gen)
            local save_name = string.format('results/category_transfer/batch%d_%s.jpg', batch_id, netT.categories[target_idx]) 
            image.save(save_name, img_montage)
            collectgarbage()
        end
    end
end

function TesterInvertion:modify_content(netT, dataloader)
-- Invert the feature map of Resnet34

    local net_hybrid = nn.Sequential()
    net_hybrid:add(self.netG)
    net_hybrid:add(netT)
    local crit_hybrid = nn.CrossEntropyCriterion()
    crit_hybrid:cuda()

    local targets = torch.Tensor(self.opt.batchSize)
    targets = targets:cuda()
    local target_idx = 102
    targets:fill(target_idx)
    local lr_code = 1000
  

   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      self.feats = self.netF:forward(self.input)
      self.input:add(1):mul(0.5)
      local codes = self.feats:clone()
      for i = 1, 1000 do
            net_hybrid:zeroGradParameters()
            local output = net_hybrid:forward(codes)
            local err = crit_hybrid:forward(output, targets)
            local df_do = crit_hybrid:backward(output, targets)
            local df_codes = net_hybrid:backward(codes, df_do)
            local norm_df = df_codes:norm()
            df_codes:div(10/norm_df)
            codes:add(df_codes:mul(-lr_code))
            local images_gen = net_hybrid.modules[1].output:clone()
            images_gen:add(1):mul(0.5)
            local string_title = string.format('%s iter %d  places2 loss=%.3f norm=%.3f', netT.categories[target_idx], i, err, norm_df) 
            print(string_title) 

            self.disp.image(images_gen,{win = self.opt.displayID, title='MODIFYING synthesized ' .. string_title})
            self.disp.image(self.input,{win = self.opt.displayID*10, title='MODIFYING real ' .. string_title})
      end
 
   end

   return lossSum / N
end

function TesterInvertion:maximize_target_tune(netT, train_opt)
    self.opt.batchSize = 12
    -- (INFO) if the output image size is 256x256, the code size should be 512x8x8
    -- (INFO) if the output image size is 224x224, the code size should be 512x7x7
    -- the hybrid net which combines the generator and the target network 
    local size_featuremap = 7
    local lr_code = train_opt.lr_code
    local start_value = train_opt.start_value


    -- the code upperbound and lowerbound (clipping the code)
    local unit_max = torch.load('pretrained/unit512_max_resnet-34.t7') -- estimated from the validation set
    unit_max = unit_max:view(1,512,1,1)
    local codes_upperbound = torch.expand(unit_max, self.opt.batchSize, 512, size_featuremap, size_featuremap)
    local codes_lowerbound = torch.zeros(self.opt.batchSize, 512, size_featuremap, size_featuremap)
    codes_upperbound = codes_upperbound:cuda()
    codes_lowerbound = codes_lowerbound:cuda()
    
    local selectIDX = {4, 23, 52, 241, 239, 90, 199, 237, 223, 376}
    for idx = 1, 10 do 
        local target_idx = selectIDX[idx]
        local category = netT.categories[target_idx] 
        local codes = torch.zeros(self.opt.batchSize, 512, size_featuremap, size_featuremap) 
        
        --codes:uniform()
        codes[{{},{},4,4}]:uniform()
        codes[{{},{},4,4}]:mul(3)

        local one_hots = torch.zeros(self.opt.batchSize, 401) -- the output of places2-alex is 401
        codes = codes:cuda()
        one_hots = one_hots:cuda()
        one_hots[{{},target_idx}]:fill(start_value) -- 5 seems the best?
    
    
        -- clip the codes before the start
        codes:cmax(codes_lowerbound)
        codes:cmin(codes_upperbound)
        for i = 1, 250 do
            self.netG:zeroGradParameters()
            netT:zeroGradParameters()
    
            local images_gen = self.netG:forward(codes)
            local max_image = torch.max(images_gen)
            local min_image = torch.min(images_gen)
            local pred_class = netT:forward(images_gen)
            
            local df_images_gen = netT:backward(images_gen, one_hots)
            -- clip the images? 
            local norm_df_images = df_images_gen:norm()
            --df_images_gen:mul(1/norm_df_images)
            --images_gen:add(df_images_gen:mul(lr_code))
    
            local df_codes = self.netG:backward(codes, df_images_gen)
    
            local norm_df_codes = df_codes:norm()
    
            --df_codes:mul(10/norm_df_codes)
            codes:add(df_codes:mul(lr_code))
            -- clip the codes during the training
            local max_code = torch.max(codes)
            local min_code = torch.min(codes)
        
            codes:cmax(codes_lowerbound)
            codes:cmin(codes_upperbound)
    
            images_gen:add(1):mul(0.5)
            self.images_gen = images_gen
            local string_title = string.format('%s iter %d  norm_dF_image=%.3f norm_df_code=%.3f max_code=%.2f min_code=%.2f max_image=%.2f min_image=%.2f lr=%.4f start_value=%d', netT.categories[target_idx], i, norm_df_images, norm_df_codes, max_code, min_code, max_image, min_image, lr_code, start_value) 
            self.disp.image(images_gen, {win = self.opt.displayID, title=string_title})
            print(string_title)
        end
        collectgarbage()
        local img_montage = montage(self.images_gen)
        local save_name = string.format('results/tune/%s_lr%.2f_start%.2f.jpg',category, lr_code, start_value) 
        image.save(save_name, img_montage)
    end
end



function TesterInvertion:maximize_target(netT, save_flag)
    self.opt.batchSize = 16
    -- (INFO) if the output image size is 256x256, the code size should be 512x8x8
    -- (INFO) if the output image size is 224x224, the code size should be 512x7x7
    -- the hybrid net which combines the generator and the target network 
    local size_featuremap = 7
    local lr_code = 0.1
    local start_value = 10
    -- the code upperbound and lowerbound (clipping the code)
    local unit_max = torch.load('pretrained/unit512_max_resnet-34.t7') -- estimated from the validation set
    unit_max = unit_max:view(1,512,1,1)
    local codes_upperbound = torch.expand(unit_max, self.opt.batchSize, 512, size_featuremap, size_featuremap)
    local codes_lowerbound = torch.zeros(self.opt.batchSize, 512, size_featuremap, size_featuremap)
    codes_upperbound = codes_upperbound:cuda()
    codes_lowerbound = codes_lowerbound:cuda()
    
    local randIDX = torch.randperm(401)

    --local gaussian_filter = image.gaussian(size_featuremap)
 --   local gaussian_filter = image.gaussian(size_featuremap, 0.15, 1)
 --   gaussian_filter:fill(1)
 --   local gaussian_filter_batch = torch.repeatTensor(gaussian_filter, self.opt.batchSize, 512, 1, 1)
 --   gaussian_filter_batch = gaussian_filter_batch:cuda()
    --local categories_select = {283, 90, 200, 369, 184}
    local categories_select = {283, 315, 70, 110, 271, 116, 174} 
    for idx = 1, #categories_select do 
        local target_idx = categories_select[idx]
        local category = netT.categories[target_idx] 
        local codes = torch.zeros(self.opt.batchSize, 512, size_featuremap, size_featuremap) 
        
        codes:uniform()
        --codes[{{},{},4,4}]:uniform()
        --codes[{{},{},4,4}]:mul(3)

        local one_hots = torch.zeros(self.opt.batchSize, 401) -- the output of places2-alex is 401
        codes = codes:cuda()
        one_hots = one_hots:cuda()
        one_hots[{{},target_idx}]:fill(10) -- 5 seems the best?
    
    
        -- clip the codes before the start
        codes:cmax(codes_lowerbound)
        codes:cmin(codes_upperbound)
        for i = 1, 150 do
            self.netG:zeroGradParameters()
            netT:zeroGradParameters()
    
            local images_gen = self.netG:forward(codes)
            local max_image = torch.max(images_gen)
            local min_image = torch.min(images_gen)
            local pred_class = netT:forward(images_gen)
            
            local df_images_gen = netT:backward(images_gen, one_hots)
            -- clip the images? 
            local norm_df_images = df_images_gen:norm()
            --df_images_gen:mul(1/norm_df_images)
            --images_gen:add(df_images_gen:mul(lr_code))
    
            local df_codes = self.netG:backward(codes, df_images_gen)
    
            local norm_df_codes = df_codes:norm()
            df_codes:cmul(gaussian_filter_batch) 
            --df_codes:mul(10/norm_df_codes)
            codes:add(df_codes:mul(lr_code))
            -- clip the codes during the training
            local max_code = torch.max(codes)
            local min_code = torch.min(codes)
        
            codes:cmax(codes_lowerbound)
            codes:cmin(codes_upperbound)
    
            images_gen:add(1):mul(0.5)
            self.images_gen = images_gen
            local string_title = string.format('%s iter %d  norm_dF_image=%.3f norm_df_code=%.3f max_code=%.2f min_code=%.2f max_image=%.2f min_image=%.2f', netT.categories[target_idx], i, norm_df_images, norm_df_codes, max_code, min_code, max_image, min_image) 
            self.disp.image(images_gen, {win = self.opt.displayID, title=string_title})
            print(string_title)
        end
        collectgarbage()
        if save_flag == true then
            local img_montage = montage(self.images_gen)
            image.save('results/places401/' .. category .. '.jpg', img_montage)
    
        end
    end
end

function TesterInvertion:computeScore(output, target, nCrops)
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

function TesterInvertion:copyInputs(sample)
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

function TesterInvertion:learningRate(epoch)
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

return M.TesterInvertion
