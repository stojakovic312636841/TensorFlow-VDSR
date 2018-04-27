import cv2
import numpy as np
import tensorflow as tf
import os 
import glob
import h5py



# Get the Image
def imread(path):
    img = cv2.imread(path)
    return img

def imsave(image, path, config):
    checkimage(image)
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(os.path.join(os.getcwd(),path),image)
    print('save image\r')



def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(1000)



def modcrop(img, scale =3):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is grayscale
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w]
    return img



def checkpoint_dir(config):
    if config.is_train:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")



def preprocess(path ,scale = 2):
    img = imread(path)

    label_ = modcrop(img, scale)
    
    # NOTE: Use to Produce the Low Resolution
    bicbuic_img = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    input_ = cv2.resize(bicbuic_img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor

    return input_, label_



def preprocess_test(path ,scale = 2):
    img = imread(path)
    
    #img is in YCrCb 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #img is only Y-channel	
    #img = img[:,:,0] 

    label_ = modcrop(img, scale)
    
    # NOTE: Use to Produce the Low Resolution
    bicbuic_img = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    input_ = cv2.resize(bicbuic_img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    
    return input_, label_


def prepare_data(dataset="Train",Input_img=""):
    """
        Args:
            dataset: choose train dataset or test dataset
            For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
    """
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), dataset) # Join the Train dir to current directory
        data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
    else:
        if Input_img !="":
            data = [os.path.join(os.getcwd(),Input_img)]
        else:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "Set5")
            data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
    return data



def load_data(is_train, test_img):
    if is_train:
        data = prepare_data(dataset="Train")
    else:
        if test_img != "":
            return prepare_data(dataset="Test",Input_img=test_img)
        data = prepare_data(dataset="Test")
    return data





def rotate(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]


    if center is None:
        center = (w / 2, h / 2)


    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def make_sub_data_train(data, config):
	"""
		Make the sub_data set
		Args:
		    data : the set of all file path 
		    config : the all flags
	"""
        sub_input_sequence = []
        sub_label_sequence = []

	for scale in range(2,5):	    

	    for i in range(len(data)):

		#input_, label_, = preprocess(data[i], config.scale) # do bicbuic only one scale
		input_, label_, = preprocess(data[i], scale) # do bicbuic turn around all scale
	
		if len(input_.shape) == 3: # is color
		    h, w, c = input_.shape
		else:
		    h, w = input_.shape # is grayscale
	
		#checkimage(input_)		

		nx, ny = 0, 0
		for x in range(0, h - config.image_size + 1, config.stride):
		    nx += 1; ny = 0
		    for y in range(0, w - config.image_size + 1, config.stride):
			ny += 1

			sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 41 * 41
			sub_label = label_[x: x + config.label_size, y: y + config.label_size] # 41 * 41


			# Reshape the subinput and sublabel
			sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
			sub_label = sub_label.reshape([config.label_size, config.label_size, config.c_dim])

			# Normialize
			sub_input =  sub_input / 255.0
			sub_label =  sub_label / 255.0
			
			#cv2.imshow("im1",sub_input)
			#cv2.imshow("im2",sub_label)
			#cv2.imshow("residual",sub_input - sub_label)
			#cv2.waitKey(0)

			# Rotate 90,180,270
			for angle in range(0,360,90):	
				sub_input = rotate(sub_input,angle)	
				sub_label = rotate(sub_label,angle)	
		
				# Add to sequence
				sub_input_sequence.append(sub_input)
				sub_label_sequence.append(sub_label)

				cv2.imshow("im1",sub_input)
				cv2.imshow("im2",sub_label)
				cv2.imshow("residual",sub_input - sub_label)
				cv2.waitKey(1)
				

        
        # NOTE: The nx, ny can be ignore in train
        return sub_input_sequence, sub_label_sequence, nx, ny



def make_sub_data_test(data, config):
    """
	Make the sub_data set
	Args:
	    data : the set of all file path 
	    config : the all flags
    """
    sub_input_sequence = []
    sub_label_sequence = []		    

    for i in range(len(data)):
	input_, label_, = preprocess_test(data[i], config.c_dim) # do bicbuic
	input_ = input_[:,:,0]
	label_ = label_[:,:,0]

	if len(input_.shape) == 3: # is color
	    h, w, c = input_.shape
	else:
	    h, w = input_.shape # is grayscale

	#checkimage(input_)	
	
	nx, ny = 0, 0
	for x in range(0, h - config.image_size + 1, config.stride):
	    nx += 1; ny = 0
	    for y in range(0, w - config.image_size + 1, config.stride):
		ny += 1

		sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 41 * 41
		sub_label = label_[x: x + config.label_size, y: y + config.label_size] # 41 * 41


		# Reshape the subinput and sublabel
		sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
		sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

		# Normialize
		sub_input =  sub_input / 255.0
		sub_label =  sub_label / 255.0
		
		#cv2.imshow("im1",sub_input)
		#cv2.imshow("im2",sub_label)
		#cv2.imshow("residual",sub_input - sub_label)
		#cv2.waitKey(10)	

		# Add to sequence
		sub_input_sequence.append(sub_input)
		sub_label_sequence.append(sub_label)
	'''
	nx, ny = 1, 1

	# Normialize
	sub_input = input_ / 255.0
	sub_label = label_ / 255.0 
  
 	sub_input_sequence.append(sub_input)
	sub_label_sequence.append(sub_label) 
	'''   
    # NOTE: The nx, ny can be ignore in train
    return sub_input_sequence, sub_label_sequence, nx, ny



def read_data(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:	
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_



def read_data_test(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('data'))
        label_ = np.array(hf.get('label'))
	
        return input_, label_



def make_data_hf(input_, label_, config):
    """
        Make input data as h5 file format
        Depending on "is_train" (flag value), savepath would be change.
    """
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

    with h5py.File(savepath, 'w') as hf:
        #checkimage(input_[1])
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)



def merge(images, size, c_dim):
    """
        images is the sub image set, merge it
    """
    h, w = images.shape[1], images.shape[2]
    
    img = np.zeros((h*size[0], w*size[1], c_dim))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h : j * h + h,i * w : i * w + w, :] = image
        #cv2.imshow("srimg",img)
        #cv2.waitKey(0)
        
    return img



def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """
    # Load data path, if is_train False, get test data
    data = load_data(config.is_train, config.test_img)
    
    # Make sub_input and sub_label, if is_train false more return nx, ny
    if config.is_train:
    	sub_input_sequence, sub_label_sequence, nx, ny = make_sub_data_train(data, config)
    else:
	sub_input_sequence, sub_label_sequence, nx, ny = make_sub_data_test(data, config)


    # Make list to numpy array. With this transform
    arrinput = np.asarray(sub_input_sequence) # [?, 41, 41, 3]
    arrlabel = np.asarray(sub_label_sequence) # [?, 41, 41, 3]
    make_data_hf(arrinput, arrlabel, config)

    return nx, ny



def input_setup_test(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """
    # Load data path, if is_train False, get test data
    data = load_data(config.is_train, config.test_img)
    
    # Make sub_input and sub_label, if is_train false more return nx, ny
    sub_input_sequence, sub_label_sequence, nx, ny = make_sub_data_test(data, config)


    # Make list to numpy array. With this transform
    arrinput = np.asarray(sub_input_sequence) # [?, 41, 41, 3]
    arrlabel = np.asarray(sub_label_sequence) # [?, 41, 41, 3]
    make_data_hf(arrinput, arrlabel, config)

    return nx, ny





def Ycbcr2RGB(Y_channel,config):
	data = load_data(config.is_train, config.test_img)
	for i in range(len(data)):
	   #input and label is YCbCr
	   input_, label_, = preprocess_test(data[i], config.scale)
	   output = np.zeros((Y_channel.shape[0], Y_channel.shape[1],3),np.uint8)
	   
	   Y_channel = Y_channel * 255
	   Y_channel = Y_channel.reshape([Y_channel.shape[0], Y_channel.shape[1]])
	  
	   output[:,:,0] = Y_channel
	   #output[:,:,0] = input_[0:Y_channel.shape[0],0:Y_channel.shape[1],0]
	   output[:,:,1] = input_[0:Y_channel.shape[0],0:Y_channel.shape[1],1]
	   output[:,:,2] = input_[0:Y_channel.shape[0],0:Y_channel.shape[1],2]
	   output = output.astype(np.uint8)
	   '''
	   for x in range(0, Y_channel.shape[0]):
	    for y in range(0, Y_channel.shape[1]):
		for z in range(0,2):
		  if output[x, y, z] > 255:
		     output[x, y, z] = 255	
		  if output[x, y, z] < 0:
		     output[x, y, z] = 0
	   '''
	   #checkimage(output[:,:,0])
	   #checkimage(output[:,:,1])	 
	   #checkimage(output[:,:,2])
	   #checkimage(output)  
	   # convert color space from bgr to rgb                          
	   output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
	return output



