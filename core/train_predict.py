import os, glob
import numpy as np
from skimage.io import imread, imsave, imshow
from PIL import Image, ImageTk
from tqdm.notebook import trange
from core.imageprep import create_crop_idx, crop_to_patch, construct_from_patch, create_crop_idx_whole

import matplotlib.pyplot as plt

def stack_predict(input_imgpath, 
                  output_imgpath, 
                  cropidx, 
                  model, 
                  rescale = None,
                  patch_size = (256, 256), 
                  predict_threshold = 0.5):
    
    IMG_HEIGHT = patch_size[0]
    IMG_WIDTH = patch_size[1]
                  
    for idx in trange(len(input_imgpath)):
        
        inputimg = input_imgpath[idx]
        
        # load image
        img_tmp = imread(inputimg)
        
        # process rescale
        if rescale is not None:  
            img_tmp = img_tmp * rescale 
        
        # crop the image
        outputimg_tmp = crop_to_patch(img_tmp, cropidx, (IMG_HEIGHT, IMG_WIDTH))
        outputimg_tmp_re = np.reshape(outputimg_tmp, (outputimg_tmp.shape[0], 
                                                      outputimg_tmp.shape[1], 
                                                      outputimg_tmp.shape[2], 1))
        
        # push the crop images into the model
        img_predict_stack = model.predict(outputimg_tmp_re, batch_size = 16, 
                                          # verbose = 1
                                         )
        
        outputimg = construct_from_patch(img_predict_stack, 
                                         cropidx, 
                                         target_size = (img_tmp.shape[0], img_tmp.shape[1]))
        
        # threshold the image
        outputimg_T = outputimg > predict_threshold
        
        # save image
        outputimg_T_pillow = Image.fromarray(outputimg_T)
        outputimg_T_pillow.save(os.path.join(output_imgpath, os.path.basename(inputimg)))

def stack_predict_v2(input_imgpath, 
                  output_imgpath, 
                  cropidx, 
                  model, 
                  rescale = None,
                  patch_size = (256, 256),
                  predict_threshold = 0.5):
    
    size_factor = 32
    
    IMG_HEIGHT = patch_size[0]
    IMG_WIDTH = patch_size[1]
    
    for idx in trange(len(input_imgpath)):
        
        inputimg = input_imgpath[idx]
        
        # load image
        img_tmp = imread(inputimg)
        
        img_height = img_tmp.shape[0]
        img_width = img_tmp.shape[1]
        
        # process rescale
        if rescale is not None:  
            img_tmp = img_tmp * rescale 
            
        # predict main region
        img_tmp_crop = img_tmp[:img_tmp.shape[0]//size_factor * size_factor, :img_tmp.shape[1]//size_factor * size_factor]
        img_tmp_crop = img_tmp_crop.reshape(1, img_tmp_crop.shape[0], img_tmp_crop.shape[1], 1)
        img_tmp_crop_predict_main = model.predict(img_tmp_crop, batch_size = 16)
        img_tmp_crop_predict_main = img_tmp_crop_predict_main.reshape(img_tmp_crop_predict_main.shape[1],
                                                            img_tmp_crop_predict_main.shape[2])
        
        ## predict the edge
        edge_patch = crop_to_patch(img_tmp, cropidx, (IMG_HEIGHT, IMG_WIDTH))
        edge_patch_re = np.reshape(edge_patch, (edge_patch.shape[0], 
                                                      edge_patch.shape[1], 
                                                      edge_patch.shape[2], 1))
        
        edge_patch_re_predict = model.predict(edge_patch_re, batch_size = 16)
        
        img_tmp_crop_predict_edge = construct_from_patch(edge_patch_re_predict, 
                                         cropidx, 
                                         target_size = (img_height, img_width))
                
        # average
        outputimg_stack = np.full((2, img_height, img_width), np.nan)
        outputimg_stack[0, :img_tmp_crop_predict_main.shape[0], :img_tmp_crop_predict_main.shape[1]] = img_tmp_crop_predict_main
        
        img_tmp_crop_predict_edge_na = img_tmp_crop_predict_edge
        img_tmp_crop_predict_edge_na[:img_height - IMG_HEIGHT, :img_width - IMG_WIDTH] = np.nan
        outputimg_stack[1, :, :] = img_tmp_crop_predict_edge
        
        outputimg = np.nanmean(outputimg_stack, axis = 0)
        
        # outputimg_T = outputimg
        
        # threshold the image
        outputimg_T = outputimg > predict_threshold
        
        
        # save image
        outputimg_T_pillow = Image.fromarray(outputimg_T)
        outputimg_T_pillow.save(os.path.join(output_imgpath, os.path.basename(inputimg)))
        
        