--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
require 'paths'
require 'nn'
require 'utils'
 
local gpuidx_main = getFreeGPU()
print('Main Process is on GPU =' .. gpuidx_main)
cutorch.setDevice(gpuidx_main)


local DataLoader = require 'dataloader'
local models = require 'models_invertion'
local TrainerInvertion = require 'train_invertion'
local opts = require 'opts_invertion'
local checkpoints = require 'checkpoints_invertion'


torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

-- Load options
local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)


-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local net, crit = models.setup(opt, checkpoint)


-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = TrainerInvertion(net, crit, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestLoss = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   
   local trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
--   local testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
--   if testLoss < bestLoss then
--      bestModel = true
--      bestLoss = testLoss
--      print(' * Best model ', testLoss)
--   end

   checkpoints.save(epoch, net, {optimState_netD = trainer.optimState_netD, optimState_netG = trainer.optimState_netG}, bestModel, opt)
end
print('finished')
