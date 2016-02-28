require "nn"
require "torch"


function BasicConvNet2()
    -- Number of filters in different layers 
    net = nn.Sequential()
    HyperColumnHeight = 193
    Layer1FilterNum = 32
    Layer2FilterNum = 64
    Layer3FilterNum = 2


    -- [[ Layer 1 ]]
    -- HyperColumnHeight input image channel, Layer1FilterNum output channels, 3x3 convolution kernel, 1 stride W, 1 stride H, 1 pad W, 1 pad R
    net:add(nn.SpatialConvolution(HyperColumnHeight, Layer1FilterNum, 3, 3, 1, 1, 1, 1)) 
    -- Batch Normalization
    net:add(nn.SpatialBatchNormalization(Layer1FilterNum))                       
    -- ReLU non-linearity
    net:add(nn.ReLU())
    -- 2x2 max-pooling
    net:add(nn.SpatialMaxPooling(2,2,2,2))


    -- [[ Layer 2 ]]
    -- Layer1FilterNum input image channel, Layer2FilterNum output channels, 3x3 convolution kernel, 1 stride W, 1 stride H, 1 pad W, 1 pad R
    net:add(nn.SpatialConvolution(Layer1FilterNum, Layer2FilterNum, 3, 3, 1, 1, 1, 1)) 
    -- Batch Normalization
    net:add(nn.SpatialBatchNormalization(Layer2FilterNum))                       
    -- ReLU non-linearity
    net:add(nn.ReLU())
    -- 2x2 max-pooling
    net:add(nn.SpatialMaxPooling(2,2,2,2))

    --[[ Layer 3 ]]
    -- Layer2FilterNum input image channel, Layer3FilterNum output channels, 3x3 convolution kernel, 1 stride W, 1 stride H, 1 pad W, 1 pad R
    net:add(nn.SpatialConvolution(Layer2FilterNum, Layer3FilterNum, 3, 3, 1, 1, 1, 1)) 
    -- Adding transfer function sigmod 
    net:add(nn.Sigmoid())  

    -- Initializing zero grad params
    net:zeroGradParameters()
    return net

end

function BasicConvNet1()
    -- Number of filters in different layers 
    net = nn.Sequential()
    HyperColumnHeight = 384
    Layer1FilterNum = 32
    Layer2FilterNum = 64
    Layer3FilterNum = 2


    -- [[ Layer 1 ]]
    -- HyperColumnHeight input image channel, Layer1FilterNum output channels, 3x3 convolution kernel, 1 stride W, 1 stride H, 1 pad W, 1 pad R
    net:add(nn.SpatialConvolution(HyperColumnHeight, Layer1FilterNum, 3, 3, 1, 1, 1, 1)) 
    -- Batch Normalization
    net:add(nn.SpatialBatchNormalization(Layer1FilterNum))                       
    -- ReLU non-linearity
    net:add(nn.ReLU())
    -- 2x2 max-pooling
    net:add(nn.SpatialMaxPooling(2,2,2,2))


    -- [[ Layer 2 ]]
    -- Layer1FilterNum input image channel, Layer2FilterNum output channels, 3x3 convolution kernel, 1 stride W, 1 stride H, 1 pad W, 1 pad R
    net:add(nn.SpatialConvolution(Layer1FilterNum, Layer2FilterNum, 3, 3, 1, 1, 1, 1)) 
    -- Batch Normalization
    net:add(nn.SpatialBatchNormalization(Layer2FilterNum))                       
    -- ReLU non-linearity
    net:add(nn.ReLU())
    -- 2x2 max-pooling
    net:add(nn.SpatialMaxPooling(2,2,2,2))

    --[[ Layer 3 ]]
    -- Layer2FilterNum input image channel, Layer3FilterNum output channels, 3x3 convolution kernel, 1 stride W, 1 stride H, 1 pad W, 1 pad R
    net:add(nn.SpatialConvolution(Layer2FilterNum, Layer3FilterNum, 3, 3, 1, 1, 1, 1)) 
    -- Adding transfer function sigmod 
    net:add(nn.Sigmoid())  

    -- Initializing zero grad params
    net:zeroGradParameters()
    return net

end


--------------------------------------------------------
-- Displays the validation results
--------------------------------------------------------
-- uses opt.num_val_images, opt.layers, opt.display_size
function display_validation_results(VGG_net, shallow_net, opt)
    local val_im_batch,val_hc_batch = create_hypercolumn_validation_dataset_bw(opt.num_val_images, VGG_net,opt.layers)
    local uv_images,y_images = create_yuv_images(val_im_batch,28,28)
    shallow_net:forward(val_hc_batch);
    local val_uv = model.output - 0.5;
    local size = opt.display_size
    print("Input, Output, Original")
    for i = 1,opt.num_val_images do
        itorch.image({image.scale(y2rgb(image.rgb2y(val_im_batch[i])),size,size),
                      image.scale(image.yuv2rgb(torch.cat(y_images[i],val_uv[i],1)),size,size),
                      image.scale(image.yuv2rgb(torch.cat(y_images[i],uv_images[i],1)),size,size)})
    end
end
