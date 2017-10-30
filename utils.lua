require 'cutorch'
local stringx = require 'pl.stringx'

function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function getFreeGPU()
    -- select the most available GPU to train
    local nDevice = cutorch.getDeviceCount()
    local memSet = torch.zeros(nDevice)
    for i=2, nDevice do
        local tmp, _ = cutorch.getMemoryUsage(i)
        memSet[i] = tmp
    end
    local _, curDeviceID = torch.max(memSet,1)
    return curDeviceID[1]
end

function readlines(fname)
    local data = file.read(fname)
    data = stringx.split(data,'\n')
    return data
end

function montage(img)
  if img:dim() == 4 or (img:dim() == 3 and img:size(1) > 3) then
    local images = {}
    for i = 1,img:size(1) do
      images[i] = img[i]
    end
    return concate_images(images, opts)
  end

end

function concate_images(images)
  local nperrow = math.ceil(math.sqrt(#images))

  local maxsize = {1, 0, 0}
  for i, img in ipairs(images) do
    if img:dim() == 2 then
      img = torch.expand(img:view(1, img:size(1), img:size(2)), maxsize[1], img:size(1), img:size(2))
    end
    images[i] = img
    maxsize[1] = math.max(maxsize[1], img:size(1))
    maxsize[2] = math.max(maxsize[2], img:size(2))
    maxsize[3] = math.max(maxsize[3], img:size(3))
  end

  -- merge all images onto one big canvas
  local numrows = math.ceil(#images / nperrow)
  local canvas = torch.FloatTensor(maxsize[1], maxsize[2] * numrows, maxsize[3] * nperrow):fill(0.5)
  local row = 0
  local col = 0
  for i, img in ipairs(images) do
    canvas:narrow(2, maxsize[2] * row + 1, img:size(2)):narrow(3, maxsize[3] * col + 1, img:size(3)):copy(img)
    col = col + 1
    if col == nperrow then
      col = 0
      row = row + 1
    end
  end

  return canvas
end
