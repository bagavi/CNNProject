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
    prob,classes = net:forward(p_img):view(-1):sort(true)

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


function create_hypercolumn_dataset(num_images, layer_nums)
    local image_dir = '../../Data/tiny-imagenet-200/test/images/'
    local max_count = num_images;
    local count = 1;
    local hc_batch = nil;
    local im_batch = nil

    for file in lfs.dir(image_dir) do
        if string.match(file, ".JPEG") then
            print( count .. ") Converting file: " .. file )
            image_path = image_dir .. file

            local im = image.load(image_path);
            local im_size = im:size();

            local hc_temp = get_VGG_hypercolumns(im,VGG_net,layer_nums)
            local hc_size = hc_temp:size();
            hc_temp = hc_temp:reshape(1,hc_size[1], hc_size[2], hc_size[3] );
            im = im:reshape(1,im_size[1],im_size[2],im_size[3])

            if count == 1 then
                hc_batch = hc_temp
                im_batch = im
            end

            hc_batch = torch.cat(hc_batch, hc_temp,1)
            im_batch = torch.cat(im_batch,im,1)

            if count == (max_count-1) then
                break;
            end
            count = count + 1;

        end
    end

    hc_dataset = {}
    hc_dataset["hypercolumns"] = hc_batch
    hc_dataset["images"] = im_batch
    return hc_dataset;
end

