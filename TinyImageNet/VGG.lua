require 'image'
require 'loadcaffe'
require "lfs"

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
    net:forward(p_img):view(-1):sort(true)

    -- layer_1 = net.modules[5].output;
    -- hyper_columns = image.scale(layer_1,224,224,'simple');
    local layer = net.modules[layer_nums[1]].output
    hyper_columns = image.scale(layer,112,112,'simple')

    --print(#layer_nums);
    for i=2,#layer_nums do
        local layer = net.modules[layer_nums[i]].output
        local scaled_layer = image.scale(layer,112,112,'simple')
        hyper_columns = torch.cat(hyper_columns,scaled_layer,1)
    end
    -- Adding the grayscale image
    scaled_image = image.scale(img,112,112,'simple')
    
    hyper_columns = torch.cat(hyper_columns, scaled_image[{{1}}], 1)
    
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

----------------------------------------------------------------------------
-- create random black&white hypercolumn dataset
----------------------------------------------------------------------------
function create_hypercolumn_dataset_random_bw(num_images, VGG_net, layer_nums)
    local max_count = num_images;
    local im_batch = get_image_batch(num_images)
    
    local count = 1;
    local y_temp = image.rgb2y(im_batch[count])
    local im_y = torch.cat(y_temp,y_temp,1);
    im_y = torch.cat(im_y,y_temp,1);
    local hc_batch = nil;
    local hc_temp = get_VGG_hypercolumns(im_y, VGG_net, layer_nums)
    local hc_size = hc_temp:size();
    local hc_batch = hc_temp:reshape(1,hc_size[1], hc_size[2], hc_size[3] );
    
    for count=2,num_images do

        y_temp = image.rgb2y(im_batch[count])
        im_y = torch.cat(y_temp,y_temp,1);
        im_y = torch.cat(im_y,y_temp,1);

        hc_temp = get_VGG_hypercolumns(im_y,VGG_net,layer_nums)
        hc_temp = hc_temp:reshape(1,hc_size[1], hc_size[2], hc_size[3] );
        hc_batch = torch.cat(hc_batch, hc_temp,1)
    end
    
    return im_batch, hc_batch
    
end


----------------------------------------------------------------------------
-- create black&white validation hypercolumn dataset
----------------------------------------------------------------------------
function create_hypercolumn_validation_dataset_bw(num_images, VGG_net, layer_nums)
    local max_count = num_images;
    local im_batch = get_validation_batch(num_images)
    
    local count = 1;
    local y_temp = image.rgb2y(im_batch[count])
    local im_y = torch.cat(y_temp,y_temp,1);
    im_y = torch.cat(im_y,y_temp,1);
    local hc_batch = nil;
    local hc_temp = get_VGG_hypercolumns(im_y, VGG_net, layer_nums)
    local hc_size = hc_temp:size();
    local hc_batch = hc_temp:reshape(1,hc_size[1], hc_size[2], hc_size[3] );
    
    for count=2,num_images do

        y_temp = image.rgb2y(im_batch[count])
        im_y = torch.cat(y_temp,y_temp,1);
        im_y = torch.cat(im_y,y_temp,1);

        hc_temp = get_VGG_hypercolumns(im_y,VGG_net,layer_nums)
        hc_temp = hc_temp:reshape(1,hc_size[1], hc_size[2], hc_size[3] );
        hc_batch = torch.cat(hc_batch, hc_temp,1)
    end
    
    return im_batch, hc_batch
    
end

----------------------------------------------------------------------------
-- create random hypercolumn dataset
----------------------------------------------------------------------------
function create_hypercolumn_dataset_random(num_images,VGG_net, layer_nums)
    local max_count = num_images;
    local im_batch = get_image_batch(num_images)
    local count = 1;
    
    local hc_batch = nil;
    local hc_temp = get_VGG_hypercolumns(im_batch[count],VGG_net,layer_nums)
    local hc_size = hc_temp:size();
    local hc_batch = hc_temp:reshape(1,hc_size[1], hc_size[2], hc_size[3] );
    
    for count=2,num_images do
        hc_temp = get_VGG_hypercolumns(im_batch[count],VGG_net,layer_nums)
        hc_temp = hc_temp:reshape(1,hc_size[1], hc_size[2], hc_size[3] );
        hc_batch = torch.cat(hc_batch, hc_temp,1)
    end
    return im_batch,hc_batch
    
end

function trim_net(net, last_layer_required)
    model_size = #net.modules-1
    for i= last_layer_required, model_size do
        net:remove(last_layer_required+1)
    end
end