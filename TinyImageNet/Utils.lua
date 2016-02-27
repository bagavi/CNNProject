require "torch"
require "image"

-- convert rgb to grayscale by averaging channel intensities
function rgb2gray(im)
    -- Image.rgb2y uses a different weight mixture

    local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
    if dim ~= 3 then
         print('<error> expected 3 channels')
         return im
    end

    -- a cool application of tensor:select
    local r = im:select(1, 1)
    local g = im:select(1, 2)
    local b = im:select(1, 3)

    local z = torch.Tensor(w, h):zero()

    -- z = z + 0.21r
    z = z:add(0.21, r)
    z = z:add(0.72, g)
    z = z:add(0.07, b)
    return z
end

-- convert grayscale to 3 channel rgb image 
function gray2rgb(grayim)
    -- Creating a local tensor variable
    width = grayim:size()[1]
    height = grayim:size()[2]
    local im = torch.Tensor(3, width, height)
    -- Initializing each channel to a gray channel
    im[1] = grayim
    im[2] = grayim
    im[3] = grayim

    return im
end

--------------------------------------------------------------
-- Returns y_images and uv_images for the batch
--------------------------------------------------------------
function create_yuv_images(images,output_w,output_h)
    w = output_w
    h = output_h
    local yuv_temp = image.rgb2yuv(images[1]);
    yuv_temp = image.scale(yuv_temp,w,h,'bilinear')
    local uv_temp = yuv_temp[{{2,3}}]
    local y_temp = yuv_temp[{{1}}]
    uv_temp = uv_temp:reshape(1,uv_temp:size()[1], uv_temp:size()[2], uv_temp:size()[3] );
    y_temp = y_temp:reshape(1,y_temp:size()[1], y_temp:size()[2], y_temp:size()[3] );
    uv_images = uv_temp
    y_images = y_temp

    for count=2,images:size()[1] do
        yuv_temp = image.rgb2yuv(images[count]);
        yuv_temp = image.scale(yuv_temp,w,h,'bilinear')
        uv_temp = yuv_temp[{{2,3}}]
        y_temp = yuv_temp[{{1}}]

        uv_temp = uv_temp:reshape(1,uv_temp:size()[1], uv_temp:size()[2], uv_temp:size()[3]);
        y_temp = y_temp:reshape(1,y_temp:size()[1], y_temp:size()[2], y_temp:size()[3] );

        uv_images = torch.cat(uv_images,uv_temp,1)
        y_images = torch.cat(y_images,y_temp,1)
    end
    return uv_images,y_images
end

--------------------------------------------------------------
-- Returns a file list of all the files in the directory
--------------------------------------------------------------
function get_file_names()
    local image_dir = '../../Data/tiny-imagenet-200/test/images/'
    local max_count = num_images;
    file_names = {};

    for file in lfs.dir(image_dir) do
        if string.match(file, ".JPEG") then
            table.insert(file_names,file)
        end
    end
    return file_names;
end


--------------------------------------------------------------
-- Returns a file list of all the files in the directory
--------------------------------------------------------------
function get_val_image_names()
    local val_image_dir = '../../Data/tiny-imagenet-200/val/images/'
    local max_count = num_images;
    file_names = {};

    for file in lfs.dir(val_image_dir) do
        if string.match(file, ".JPEG") then
            table.insert(file_names,file)
        end
    end
    return file_names;
end

--------------------------------------------------------------
-- Returns a random batch of images
--------------------------------------------------------------
function get_image_batch(num_images)
    local image_dir = '../../Data/tiny-imagenet-200/test/images/'
    local max_count = num_images;
    local count = 1;
    local im_batch = nil
    local file_names = get_file_names();
    local num_files = #file_names
    
    -- Bad code
    local file = file_names[1];
    local image_path = image_dir .. file
    local im = image.load(image_path);
    local im_size = im:size();
    local im_batch = torch.Tensor(num_images, im_size[1], im_size[2], im_size[3])
    
    math.randomseed( os.time() )
    for count=1,num_images do
        rand_index = math.random(1,num_files);
        
        local file = file_names[rand_index];
        local image_path = image_dir .. file
        local im = image.load(image_path,3);
        local im_size = im:size();
        im_batch[count] = im:reshape(1,im_size[1],im_size[2],im_size[3])
        -- else
        --     count = count -1
        -- end    
    end
    return im_batch
end



--------------------------------------------------------------
-- Returns the validation batch
--------------------------------------------------------------
function get_validation_batch(num_images)
    local val_image_dir = '../../Data/tiny-imagenet-200/val/images/'
    local max_count = num_images;
    local count = 1;
    local im_batch = nil
    local file_names = get_val_image_names();
    local num_files = #file_names
    
    -- Bad code
    local file = file_names[1];
    local image_path = val_image_dir .. file
    local im = image.load(image_path);
    local im_size = im:size();
    local im_batch = torch.Tensor(num_images, im_size[1], im_size[2], im_size[3])
    
    -- IMP code
    math.randomseed(10)
    for count=1,num_images do
        rand_index = math.random(1,num_files);
        local file = file_names[rand_index];
        local image_path = val_image_dir .. file
        local im = image.load(image_path,3);
        local im_size = im:size();
        im_batch[count] = im:reshape(1,im_size[1],im_size[2],im_size[3])
        -- else
        --     count = count -1
        -- end    
    end
    return im_batch
end

--------------------------------------------------------
-- Stack 3 intensity y channels to make rgb
--------------------------------------------------------
function y2rgb(y_temp)
    im_y = torch.cat(y_temp,y_temp,1);
    im_y = torch.cat(im_y,y_temp,1);
    return im_y
end

