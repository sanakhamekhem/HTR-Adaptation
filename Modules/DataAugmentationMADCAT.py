#!/usr/bin/env python3

"""
    `Modules implementing affine and morphological distortions to images in
    numpy datatype.
"""
from PIL import Image
import numpy as np


from PIL import Image, ImageChops

def remove_padding(img):
	img=255-img
	image = Image.fromarray(img)
	bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
	diff = ImageChops.difference(image, bg)
	bbox = diff.getbbox()
	if not bbox:
		print('not sa')
		return image

	return image.crop(bbox)



def scale(img,scale_factor ):
	imgPIL = Image.fromarray(img)
	ho, vo = imgPIL.size

	hn, vn = int(scale_factor*ho), int(scale_factor*vo)
	img_sc = imgPIL.resize((hn, vn))

	img = np.array(img_sc).reshape((vn,hn))
			
	return img

	
def rotate(img,rotate_angle,rotate_prob = 0.1, rotate_prec = 100):
 
	import cv2
	rows,cols = img.shape
	print cols
	print rows
	rotate_prec = rotate_prec * max(rows/cols, cols/rows)

	M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_angle,1)
	img = cv2.warpAffine(img,M,(cols,rows))
	return img 
	

def translate(img, h_translation_factor, v_translation_factor):
 
	import cv2
	rows,cols = img.shape

	M = np.float32([[1,0,h_translation_factor],[0,1,v_translation_factor]])
	img = cv2.warpAffine(img,M,(cols,rows))

	return img
	

def dilate(img,n1,n2):
    

	import cv2

	kernel = np.ones((n1,n2), np.uint8) 
	img = cv2.dilate(img,kernel,iterations = 1)

	return img

def erode(img,n1,n2):
 	import cv2
	# Taking a matrix of size 5 as the kernel 
	kernel = np.ones((n1,n2), np.uint8) 
	img=255-img
	img = cv2.erode(img,kernel,iterations = 1)
	img=255-img
	return img



def transf1(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = rotate(img_np,0.2)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)
	
def transf2(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = translate(img_np, 0.2, 0.3)
	img_np = dilate(img_np, 2,2)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)	
	
 
def transf4(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.8)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)	

def transf5(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.8)
	img_np = translate(img_np, 0.2, 0.3)
	img_np = rotate(img_np,0.2)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf5(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.8)
	img_np = rotate(img_np,0.2)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf6(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.8)
	img_np = rotate(img_np,0.1)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)
def transf7(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = dilate(img_np, 1,1)
	img_np = rotate(img_np,0.3)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf8(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.5)
	img_np = translate(img_np, 0.2, 0.3)
	img_np = rotate(img_np,0.25)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf9(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = translate(img_np, 0.2, 0.3)
	img_np = rotate(img_np,0.1)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf10(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.7)
	img_np = translate(img_np, 0.1, 0.3)
	img_np = dilate(img_np, 2,2)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)


def transf11(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.4)
	img_np = rotate(img_np,0.1)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf12(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.3)
	img_np = rotate(img_np,0.2)

	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)
def transf13(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.3)
	img_np = rotate(img_np,0.1)

	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf14(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.45)
	img_np = rotate(img_np,0.2)
	
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)
def transf10(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = dilate(img_np, 2,2)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)
def transf3(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	img_np=255-img_np
	img_np = scale(img_np, 0.8)
	img_np = translate(img_np, 0.2, 0.3)
	img_np =dilate(img_np, 1,1)
	img_np=255-img_np
	ret2,img_np = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite(destfile1, img_np)

def transf0(sourcefile,destfile1,origfname):
	tmppath='RCNN/HTRTF/tmp/'
	import cv2
	img = Image.open(sourcefile)
	img.save(tmppath + origfname)

	img =  cv2.imread(tmppath + origfname,0)
	 
	img_np = img
	
	cv2.imwrite(destfile1, img_np)	

def distort(img_list):
    new_list=[]
    for ind, img_np in enumerate(img_list):
        #img_np = 255 - img_np
        img_np = translate(img_np, 0.2, 0.3)
        img_np = rotate(img_np,0.2)
        img_np = shear(img_np)
        img_np = scale(img_np, 0.8)
        img_np = dilate(img_np)
        img_np = erode(img_np,4,4)
        new_list.append(img_np)
    
    return new_list
	
def augment(liste): 
	filelist='RCNN/HTRTF/Projects/MADCAT-WriterDep-Global/Set/' + liste
	destpath='RCNN/augmented-MADCAT/'
	c=1
	with open(filelist) as f:
		for l in f:
			c=1
			l=l.replace('\r', '')
			l=l.replace('\n', '')
			fname=l + '.png'
			origfname=fname
			sourcefile='RCNN/imagesMADCAT/' + fname
			destfile0=destpath + l +'.png'
			destfile1=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile2=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile3=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile4=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile5=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile6=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile7=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile8=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile9=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile10=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile11=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile12=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile13=destpath + l + '_' + str(c) +'.png'
			c=c+1
			destfile14=destpath + l + '_' + str(c) +'.png'
			c=c+1
			transf0(sourcefile,destfile0,origfname)
			transf1(sourcefile,destfile1,origfname)
			transf4(sourcefile,destfile2,origfname)
			#ttransf3(sourcefile,destfile3,origfname)
			#ttransf4(sourcefile,destfile4,origfname)
			#ttransf5(sourcefile,destfile5,origfname)
			#ttransf6(sourcefile,destfile6,origfname)
			#ttransf7(sourcefile,destfile7,origfname)
			#ttransf8(sourcefile,destfile8,origfname)
			#ttransf9(sourcefile,destfile9,origfname)
			#ttransf10(sourcefile,destfile10,origfname)
			#ttransf11(sourcefile,destfile11,origfname)
			#ttransf12(sourcefile,destfile12,origfname)
			#ttransf13(sourcefile,destfile13,origfname)
			#transf14(sourcefile,destfile14,origfname)


def main():
	augment('list_train')
	augment('list_valid')

	
if __name__ == '__main__':
    main()
