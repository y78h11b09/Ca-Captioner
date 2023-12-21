import cv2
import numpy as np
from PIL import Image, ImageFilter
# def add_gaussian_noise(image):
#     row ,col,ch = image.shape
#     mean = 0
#     var = 0.01
#     sigma = var**0.5
#     gauss =np.random.normal(mean,sigma,(row,col,ch))
#     gauss = gauss.reshape(row,col,ch)
#     noisy = image+gauss
#     # noisy = np.clip(noisy,0.0,255.0)
#     return noisy
#
# imag_path = '/media/huashuo/mydisk/yang/Base1/data/image_datang_resized2/1.jpg'
# original_image = cv2.imread(imag_path)
#
# noisy_image = add_gaussian_noise(original_image)
# # noisy_image = np.clip(noisy_image,0,1)
# noisy_image = noisy_image.astype(np.uint8)
# noisy_image = Image.fromarray(noisy_image)
#
# output_path = '/media/huashuo/mydisk/yang/Base1/data/image_datang_resized2/noise.jpg'
# noisy_image.save(output_path)
# noisy_image.show()
# cv2.imshow('original_image', original_image)
# cv2.imshow('noisy_image',noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #
import skimage
from skimage import io, util,img_as_ubyte
origin = skimage.io.imread('/media/huashuo/mydisk/yang/Base1/data/image_datang_resized2/2.jpg')
origin = img_as_ubyte(origin)
noisy = util.random_noise(origin,mode='gaussian',var=0.1,clip=True)
noisy1 = util.random_noise(origin,mode='s&p',amount=0.65,clip=True)
# noisy = origin+noisy
noisy = img_as_ubyte(noisy)
# cv2.imshow('noisy_image',noisy)
noisy1 = img_as_ubyte(noisy1)
output_path = '/media/huashuo/mydisk/yang/Base1/data/image_datang_resized2/noise.jpg'
output_path1 = '/media/huashuo/mydisk/yang/Base1/data/image_datang_resized2/noise2.jpg'
# io.imsave(output_path,noisy)
# io.imshow(noisy)
io.imsave(output_path1,noisy1)
io.imshow(noisy1)


io.show()