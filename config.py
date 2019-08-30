class Config:	
    #Store the address of the voc data set
    VOC_ROOT = '/Users/liuhaoxin/data/VOC/VOCdevkit'

    #Number of categories classified (including background count)
    num_classes = 21

    #If you want to continue training from the already trained model,uncomment the next line and 
    #fill in the name of the model you continue to train.If you train from the beginning, stay the same.
    #resume = 'weights/loss-2079.08.pth'
    resume = None
    
    #Below are some of the parameters of the commonly used model training, which are changed as needed. 
    #Learning rate, too high learning rate may lead to loss of nan
    lr = 0.001
    #Batch size size, too large may cause explosion memory
    batch_size = 32 
    
    #The following two parameters are generally not needed for gradient acceleration and over-fitting prevention.
    momentum = 0.9
    weight_decay = 5e-4

    #How many epcoch to train
    epoch = 100 

    #These two parameters can be adjusted, lr_reduce_epoch is how many epoch learning rate multiplied by gamma
    #For example, gamma is 0.2, lr_reduce_epoch=30, ie every 30 epoch, the learning rate is reduced by 5 times.
    gamma = 0.2
    lr_reduce_epoch = 30

    #Folders saved by the model, as well as pre-trained network weights
    save_folder = 'weights/'
    basenet = 'vgg16_reducedfc.pth'

    #How many batches of training are printed once per message
    log_fn = 10 

    #Negative sample ratio
    neg_radio = 3  

    #Some parameters in ssd, the following parameters are based on the paper, carefully modified! !
    #Enter the size of the image
    min_size = 300
    #There are 6 feature layers in ssd, and their sizes are as follows for the generation of anchor.
    grids = (38, 19, 10, 5, 3, 1)
    #Proportion of scaling for each feature layer anchor
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    #How many times the original image size (300) is reduced compared to the feature layer, such as (300 / 38 is approximately equal to 8)
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    #The size of the anchor prevented on each feature layer is, in principle, the smaller the feature map, the larger the placement.
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)] 
    #The number of anchors on each feature map, used to generate the channel of the last layer at the time of model construction
    anchor_num = [4, 6, 6, 6, 4, 4]

    #The mean value above each channel of the picture
    mean = (104, 117, 123)
    #Variance for coordinate coding
    variance = (0.1, 0.2)

opt = Config()
