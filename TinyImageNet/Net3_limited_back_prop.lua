require 'Utils'
require 'nngraph';
require 'loadcaffe'
require 'image'
require 'VGG'
require 'optim'

function create_colorNet() 
    local net = load_VGG();
    
    local input = nn.Identity()():annotate{
       name = 'Conv Layer 1', description = 'Image Features',
       graphAttributes = {color = 'green'}
    }
    local conv1  = net.modules[1]
    local conv3  = net.modules[3]
    local conv6  = net.modules[6]
    local conv8  = net.modules[8]
    local conv11 = net.modules[11]
    local conv13 = net.modules[13]
    local conv15 = net.modules[15]
    local conv18 = net.modules[18]
    local conv20 = net.modules[20]
    local conv22 = net.modules[22]
--     local conv25 = net.modules[25]
--     local conv27 = net.modules[27]
--     local conv29 = net.modules[27]
--     local FC33   = net.modules[33]
--     local FC36   = net.modules[36]
--     local FC39   = net.modules[39]

    local level_0  = conv1(input):annotate{
       name = 'Conv Layer 1', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    local level_1  = conv3(nn.ReLU()(level_0)):annotate{
       name = 'Conv Layer 2', description = 'Image Features',
       graphAttributes = {color = 'blue'}
    }
    local level_2  = conv6(nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()(level_1))):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    local level_3  = conv8(nn.ReLU()(level_2)):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'blue'}
    }
    local level_4  = conv11( nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()(level_3)) ):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    local level_5  = conv13(nn.ReLU()(level_4)):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    local level_6  = conv15(nn.ReLU()(level_5)):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'blue'}
    }
    
    -- New addition.
    local level_7  = conv18( nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()(level_6)) ):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    local level_8  = conv20(nn.ReLU()(level_7)):annotate{
       name = 'Conv Layer 4', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    local level_9  = conv22(nn.ReLU()(level_8)):annotate{
       name = 'Conv Layer 4', description = 'Image Features',
       graphAttributes = {color = 'blue'}
    }
    
    ----------------------------------------------------------------------------------------------
    
    local dimension = 2
    -- Deconvoluting level 3 and level 6
    local level_3_deconv = nn.SpatialFullConvolution(128, 32, 1, 1, 2, 2, 0, 0, 1, 1)(level_3):annotate{
       name = 'Deconving level 3', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }
    local level_6_deconv = nn.SpatialFullConvolution(256, 32, 2, 2, 4, 4, 0, 0, 2, 2)(level_6):annotate{
       name = 'Deconving level 6', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }
    
    local level_9_deconv = nn.SpatialFullConvolution(512, 64, 4, 4, 8, 8, 0, 0, 4, 4)(level_9):annotate{
       name = 'Deconving level 6', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }
    
    local output_VGG = (nn.JoinTable(dimension)({input, 
                                                level_1, 
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

    local level_8 = nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()( level_7point5 ))
    
    local level_9= nn.Tanh()( nn.SpatialConvolution(128, 2, 3, 3, 1, 1, 1, 1)(level_8)):annotate{
       name = 'Final Layer', description = 'Final output. Using Sigmoid',
       graphAttributes = {color = 'purple'}
    }


-- For 1x1 conv layer
--     local level_6point5 = nn.SpatialBatchNormalization(32)( nn.SpatialConvolution(131, 32, 1, 1, 1, 1, 0, 0)(output_VGG))
--     local level_7point5 = nn.SpatialBatchNormalization(64)( nn.SpatialConvolution(32, 64, 1, 1, 1, 1, 0, 0)(level_7))
--     local level_9= nn.Tanh()( nn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0)(level_8)):annotate{
--        name = 'Final Layer', description = 'Final output. Using Sigmoid',
--        graphAttributes = {color = 'purple'}

    model = nn.gModule({input}, {level_9})
    -- Save the networks graph

    -- Back Propogate only till layer 15
    -- Restricting the backwardnodes of the model to accomplish the above
    
    all_bn = model.bg:topsort() -- Contains all the backward nodes
    model.backwardnodes = { all_bn[1],
                            all_bn[2],
                            all_bn[3],
                            all_bn[4],
                            all_bn[5],
                            all_bn[6],
                            all_bn[7],
                            all_bn[8],
                            all_bn[9],
                            all_bn[10],
                            all_bn[11],
                            all_bn[12],
                            all_bn[13],
                            all_bn[14],
                            all_bn[15],
                        }
    -- Changing the color of the nodes which have will have backpropogation
    
--     for i=1,15 do
--         all_bn[i].data.annotations.graphAttributes.color = 'green'
--     end
    
    graph.dot(model.bg, 'MLP', 'VGGnet')

    ----------------------------------------------------------------------------------------------
    
    return model  
end    

