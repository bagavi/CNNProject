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