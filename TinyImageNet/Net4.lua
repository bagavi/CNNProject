require 'Utils'
require 'nngraph';
require 'loadcaffe'
require 'image'
require 'VGG'
require 'optim'

function create_colorNet() 
    local dtype = 'torch.FloatTensor'
    
    -- Input image
    local input1 = nn.Identity()():annotate{
       name = 'Conv Layer 1', description = 'Input Layer',
       graphAttributes = {color = 'green'}
    }
    -- Features from layer 3
     local input2 = nn.Identity()():annotate{
       name = 'Conv Layer 2', description = 'Image Features',
       graphAttributes = {color = 'green'}
    }

    -- Features from layer 6
    local input3 = nn.Identity()():annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'green'}
    }

    -- Features from layer 9    
     local input4 = nn.Identity()():annotate{
       name = 'Conv Layer 4', description = 'Image Features',
       graphAttributes = {color = 'green'}
    }



    ----------------------------------------------------------------------------------------------

    local dimension = 2
    -- Deconvoluting level 3 and level 6
    local level_3_deconv = nn.SpatialFullConvolution(128, 32, 1, 1, 2, 2, 0, 0, 1, 1)(input2):annotate{
       name = 'Deconving level 3', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }
    local level_6_deconv = nn.SpatialFullConvolution(256, 32, 2, 2, 4, 4, 0, 0, 2, 2)(input3):annotate{
       name = 'Deconving level 6', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }

    local level_9_deconv = nn.SpatialFullConvolution(512, 64, 4, 4, 8, 8, 0, 0, 4, 4)(input4):annotate{
       name = 'Deconving level 6', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }

    local output_VGG = (nn.JoinTable(dimension)({input1, 
                                                level_3_deconv, 
                                                level_6_deconv, 
                                                level_9_deconv})):annotate{
       name = 'Joining layer. Fuck yeah!', description = 'Joining input, level1, deconved_level3, deconved_level6',
       graphAttributes = {color = 'grey'}
    }


    -- Building our own network from here. 3 layers
    local level_6point5 = nn.SpatialBatchNormalization(64)( nn.SpatialConvolution(195, 64, 3, 3, 1, 1, 1, 1)(output_VGG))
    local level_7 = nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()( level_6point5 ))

    local level_7point5 = nn.SpatialBatchNormalization(128)( nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)(level_7))

    local level_7point5_2 = nn.SpatialBatchNormalization(128)( nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(level_7point5))

    local level_8 = nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()( level_7point5_2 ))

    local level_9= nn.Tanh()( nn.SpatialConvolution(128, 2, 3, 3, 1, 1, 1, 1)(level_8)):annotate{
       name = 'Final Layer', description = 'Final output. Using Sigmoid',
       graphAttributes = {color = 'purple'}
    }

     model = nn.gModule({input1, input2,input3, input4}, {level_9})
    ----------------------------------------------------------------------------------------------
    
    return model  
end    

