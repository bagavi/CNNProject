require 'Utils'
require 'nngraph';
require 'loadcaffe'
require 'image'
require 'VGG'
require 'optim'

function create_colorNet() 
    net = load_VGG();
    
    input = nn.Identity()():annotate{
       name = 'Conv Layer 1', description = 'Image Features',
       graphAttributes = {color = 'green'}
    }
    conv1  = net.modules[1]
    conv3  = net.modules[3]
    conv6  = net.modules[6]
    conv8  = net.modules[8]
    conv11 = net.modules[11]
    conv13 = net.modules[13]
    conv15 = net.modules[15]
    conv18 = net.modules[18]
    conv20 = net.modules[20]
    conv22 = net.modules[22]
    conv25 = net.modules[25]
    conv27 = net.modules[27]
    conv29 = net.modules[27]
    FC33   = net.modules[33]
    FC36   = net.modules[36]
    FC39   = net.modules[39]

    level_0  = conv1(input):annotate{
       name = 'Conv Layer 1', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    level_1  = conv3(nn.ReLU()(level_0)):annotate{
       name = 'Conv Layer 2', description = 'Image Features',
       graphAttributes = {color = 'blue'}
    }
    level_2  = conv6(nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()(level_1))):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    level_3  = conv8(nn.ReLU()(level_2)):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'blue'}
    }
    level_4  = conv11( nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()(level_3)) ):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    level_5  = conv13(nn.ReLU()(level_4)):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'red'}
    }
    level_6  = conv15(nn.ReLU()(level_5)):annotate{
       name = 'Conv Layer 3', description = 'Image Features',
       graphAttributes = {color = 'blue'}
    }
    ----------------------------------------------------------------------------------------------
    
    dimension = 2
    -- Deconvoluting level 3 and level 6
    level_3_deconv = nn.SpatialFullConvolution(128, 16, 1, 1, 2, 2, 0, 0, 1, 1)(level_3):annotate{
       name = 'Deconving level 3', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }
    level_6_deconv = nn.SpatialFullConvolution(256, 16, 2, 2, 4, 4, 0, 0, 2, 2)(level_6):annotate{
       name = 'Deconving level 6', description = 'To increase the dimension',
       graphAttributes = {color = 'yellow'}
    }
    output_VGG = (nn.JoinTable(dimension)({input, level_1, level_3_deconv, level_6_deconv})):annotate{
       name = 'Joining layer', description = 'Joining input, level1, deconved_level3, deconved_level6',
       graphAttributes = {color = 'grey'}
    }


    -- Building our own network from here. 3 layers
    level_6point5 = nn.SpatialBatchNormalization(32)( nn.SpatialConvolution(99, 32, 3, 3, 1, 1, 1, 1)(output_VGG))
    level_7 = nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()( level_6point5 ))

    level_7point5 = nn.SpatialBatchNormalization(64)( nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)(level_7))
    level_8 = nn.SpatialMaxPooling(2,2,2,2)(nn.ReLU()( level_7point5 ))

    level_9= nn.Sigmoid()( nn.SpatialConvolution(64, 2, 3, 3, 1, 1, 1, 1)(level_8)):annotate{
       name = 'Final Layer', description = 'Final output. Using Sigmoid',
       graphAttributes = {color = 'purple'}
    }

    model = nn.gModule({input}, {level_9})
    --netsav = model:clone('weight', 'bias', 'running_mean', 'running_std')

    graph.dot(model.fg, 'MLP', 'VGGnet')

    ----------------------------------------------------------------------------------------------
    
    return model  
end    

