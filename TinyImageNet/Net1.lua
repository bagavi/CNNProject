require "nn"
require "torch"

local function BasicConvNet1()
	-- Number of filters in different layers 
	net = nn.Sequential()
	HyperColumnHeight = 512
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