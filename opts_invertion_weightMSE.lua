--
--
local M = { }

function M.parse(arg)

-- Options:
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Feature Invertion from ResNet Training script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '/data/vision/torralba/deepscene/places365_standard',         'Path to dataset')
   cmd:option('-dataset',    'places365', 'Options: imagenet | cifar10 | cifar100 | places365')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        5, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         100,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       48,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.01,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   cmd:option('-LR_gan_netD', 0.0002, 'lr for GAN discriminator')
   cmd:option('-beta_gan_netD', 0.5)
   cmd:option('-LR_gan_netG', 0.00005, 'lr for GAN generator')
   cmd:option('-beta_gan_netG', 0.7)

   ---------- Model options ----------------------------------
   cmd:option('-netType',      'resnet', 'Options: resnet | preresnet')
   cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', '',       'Options: A | B | C')
   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   
   ---------- Feature Invertion options --------------------
   cmd:option('-invertion', 1, 'Flag for doing feature invertion')
   cmd:option('-display', 1, 'if show up in display library')
   -- loss function
   cmd:option('-loss_l2', 1, 'whether to include the image loss')
   cmd:option('-factor', 0.2, 'the magic number to get high frequency component')

   cmd:option('-model_pretrain', 'pretrained/resnet-34.t7', 'the feature extraction network')
   cmd:option('-type_netG', 3, 'the type of the generator network')
   cmd:option('-type_netD', 1, 'the type of the discriminator network')
   cmd:option('-nz', 512,  'the dimensionality of the deep feature')
   cmd:option('-ngf', 64, '# of gen filters in first conv layer')
   cmd:option('-nc', 3, '# of output channel')
   cmd:option('-ndf', 64, '# of discriminator filters in first conv layer')	

 
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.exp_name = ''
   opt.displayID = opt.factor * 10
   ---------

   return opt
end

return M
