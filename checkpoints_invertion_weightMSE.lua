--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil, {}
   end

   local latestPath = paths.concat(opt.resume, opt.exp_name .. '_latest.t7')
   if not paths.filep(latestPath) then
      return nil, {}
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState
end

function checkpoint.save(epoch, nets, optimState, isBestModel, opt)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network

   nets.netG = deepCopy(nets.netG):float():clearState()
   nets.netD = deepCopy(nets.netD):float():clearState()

   local modelFile = opt.exp_name .. '_model_' .. epoch .. '.t7'
   local optimFile = opt.exp_name .. '_optimState_' .. epoch .. '.t7'
   print('=> saving model:' .. modelFile)
   print('=> saving optimFile:' .. optimFile)
   torch.save(paths.concat(opt.save, modelFile), {netG=nets.netG, netD = nets.netD})
   torch.save(paths.concat(opt.save, optimFile), {optimState_netD = optimState.optimState_netD, optimState_netG = optimState.optimState_netG})
   torch.save(paths.concat(opt.save, opt.exp_name .. '_latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if isBestModel then
      torch.save(paths.concat(opt.save, opt.exp_name .. '_model_best.t7'), {netG=nets.netG, netD = nets.netD})
   end
end

return checkpoint
