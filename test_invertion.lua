require 'paths'
require 'nn'
require 'utils'
require 'image'


local gpuidx_main = getFreeGPU()
print('Main Process is on GPU =' .. gpuidx_main)
cutorch.setDevice(gpuidx_main)

local models = require 'models_invertion'
local DataLoader = require 'dataloader'
local opts = require 'opts_invertion'
local TesterInvertion = require 'tester_invertion'
local checkpoints = require 'checkpoints_invertion'



torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

-- Load options
local opt = opts.parse(arg)
opt.test = 1 -- test state
opt.resume = 'checkpoints'
opt.modelFile = 'save/l2GANfeat_model_3.t7'
opt.resnet_type = 'conv'
cutorch.manualSeedAll(opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- Load the pretrained model
local checkpoint = {}
local optimState = {}
checkpoint.modelFile = opt.modelFile
local net, crit = models.setup(opt, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
local tester = TesterInvertion(net, crit, opt, optimState)

-- Start inverting the batch
local function_id = 1


if function_id == 1 then
    print('Start inverting the image batches given the validation set of places2......')
    local trainLoader, valLoader = DataLoader.create(opt)
    tester:invert_batch(0, valLoader)

elseif function_id ==  2 then
    print('Start activation maximization.......')
    local filename_netT = 'pretrained/places2_alexnet.t7'
    local categories = readlines('pretrained/categories_places2_convert.txt')
    local netT = torch.load(filename_netT)
    netT:evaluate()
    netT:cuda()
    netT.categories = categories

    if test_true == 1 then
        -- test the pretrained model
        local test_img = image.load('pretrained/7.jpg')
        test_img = image.scale(test_img, 224, 224)
        test_img = test_img:cuda()
        local prob = netT:forward(test_img)
        local y, i = torch.sort(prob[1], 1, true) 
        for idx = 1, 5 do
            print(categories[i[idx]])
        end
    end
    tester:maximize_target(netT)

elseif function_id == 9 then
    -- caffe model maximization generation

elseif function_id == 8 then
    -- guided generation 
    print('Guided generation, given the validation set of places2......')
    local filename_netT = 'pretrained/places2_alexnet.t7'
    local categories = readlines('pretrained/categories_places2_convert.txt')
    local netT = torch.load(filename_netT)
    netT:evaluate()
    netT:cuda()
    netT.categories = categories
    opt.input_size = 224
    opt.batchSize = 49
    local trainLoader, valLoader = DataLoader.create(opt)
    tester:guided_generation(netT, valLoader)

elseif function_id ==  10 then
    print('Start activation maximization (parameter searching).......')
    local filename_netT = 'pretrained/places2_alexnet.t7'
    local categories = readlines('pretrained/categories_places2_convert.txt')
    local netT = torch.load(filename_netT)
    netT:evaluate()
    netT:cuda()
    netT.categories = categories

    local lr_code_set = {0.1, 0.5, 1, 0.01, 0.001}
    local start_value_set = {1, 3, 5, 10 ,20 ,30, 50}
    for i = 1, #lr_code_set do
        for j = 1, #start_value_set do
            local tune_opt = {}
            tune_opt.lr_code = lr_code_set[i]
            tune_opt.start_value = start_value_set[j]
            tester:maximize_target_tune(netT, tune_opt)
        end
    end


elseif function_id == 3 then
    print('Start modifying the image content')
    local filename_netT = 'pretrained/places2_alexnet.t7'
    local categories = readlines('pretrained/categories_places2_convert.txt')
    local netT = torch.load(filename_netT)
    netT:evaluate()
    netT:cuda()
    netT.categories = categories

    opt.input_size = 224 -- the input should be 224 otherwise places2 alexnet will complain
    local trainLoader, valLoader = DataLoader.create(opt)
    tester:modify_content(netT, valLoader)


elseif function_id== 4 then
    print('Start getting the statistics of the code......')
    opt.input_size = 256
    local trainLoader, valLoader = DataLoader.create(opt)
    tester:getStat_code(valLoader)



--
else
    print('nothing to do')
end


