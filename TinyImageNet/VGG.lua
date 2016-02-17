require 'image'
require 'loadcaffe'

-- Loads the mapping from net outputs to human readable labels
function load_synset()
  local list = {}
  for line in io.lines'synset_words.txt' do
    table.insert(list, string.sub(line,11))
  end
  return list
end

function preprocess(img) 
    -- 16 layer VGG expects a 3x224x224 sized image
  img = image.scale(img, 224, 224)
-- Directly obtained from the website
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
-- Permuting from RBG to BGR
  local perm = torch.LongTensor{3, 2, 1}
-- Scaling the elements from 0:1 to 0:256
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
-- Subtracting the mean
  img:add(-1, mean_pixel)
  return img
end

function VGG_forward_test(img,net)
    p_img = preprocess(img)
    prob,classes = net:forward(p_img):view(-1):sort(true)
    synset_words = load_synset()
    classes5 = {}
    for i=1,5 do
      table.insert(classes5, synset_words[classes[i]])
    end

    return classes5
end

------------------------------------------------------------------
-- Returns the VGG hyperolumns
-----------------------------------------------------------------
function get_VGG_hypercolumns(img,net,layer_nums)
    p_img = preprocess(img)
    prob,classes = net:forward(p_img):view(-1):sort(true)

    -- layer_1 = net.modules[5].output;
    -- hyper_columns = image.scale(layer_1,224,224,'simple');
    local layer = net.modules[layer_nums[1]].output
    hyper_columns = image.scale(layer,112,112,'simple')

    print(#layer_nums);
    for i=2,#layer_nums do
        local layer = net.modules[layer_nums[i]].output
        local scaled_layer = image.scale(layer,112,112,'simple')
        hyper_columns = torch.cat(hyper_columns,scaled_layer,1)
    end

    return hyper_columns
end
----------------------------------------------------------------------------
-- Loads a pretrained VGG. trained caffe binary, prototxt mentioned 
-- in the function.
----------------------------------------------------------------------------

function load_VGG()
    prototxt = '../../Data/VGG_caffe/VGG_ILSVRC_16_layers_deploy.prototxt'
    binary = '../../Data/VGG_caffe/VGG_ILSVRC_16_layers.caffemodel'

    -- this will load the network and print it's structure
    net = loadcaffe.load(prototxt, binary);
    return net
end

