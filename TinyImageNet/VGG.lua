require 'image'

-- Loads the mapping from net outputs to human readable labels
function load_synset()
  local list = {}
  for line in io.lines'synset_words.txt' do
    table.insert(list, string.sub(line,11))
  end
  return list
end

function preprocess(img) 
   require 'image'
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

function VGG_forward( image, net )
--    itorch.image(image)
    image = preprocess(image)
    prob, classes = net:forward(image):view(-1):sort(true)
--    synset_words = load_synset()
--    classes5 = torch.Tensor(5)
--    for i=1,5 do
--      classes5[i] =  synset_words[classes[i]])
--    end
    return image
end