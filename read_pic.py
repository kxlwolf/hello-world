# -*- coding: utf-8 -*-

import os
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import skimage
import cv2
import matplotlib.pyplot as plt


"""PIL.Image.open 不直接返回numpy对象，可以用numpy提供的函数进行转换"""
def PIL_read():
    img_name = "G:\Chinobot\programs\python\image-similarity\images\save_bbox\DJI_0031360_3.jpg"
    img = Image.open(img_name)      
    img_array = np.array(img)
    img.show()
    img.save("./PIL_save.jpg")
    pixel = img.getpixel((7, 7))
#    for pixel in img.getdata():
#        print("pixel: ", pixel)  
        
    print("img.mode: ", img.mode)
    #print("img.dtype: ", img.dtype)  # no dtype  
    #print("type(img): ", type(img))  # no type
    print("img.size: ", img.size)
    #print("img.shape: ", img.shape)
    print("img: ", img)
    
    print("img_array.dtype: ", img_array.dtype)
    print("type(img_array): ", type(img_array))
    print("img_array.size: ", img_array.size)
    print("img_array.shape: ", img_array.shape)
    print("img_array: ", img_array)
    print("img_array[img_array > 0]: ", img_array[img_array > 0])
        
    img_width = img_array.shape[1]
    img_height = img_array.shape[0]
    print("img_width: ", img_width, "img_height: ", img_height)

    
    img_gray = img.convert('L')
    print("img_gray: ", img_gray)
    print("img_gray_array: ", np.array(img_gray))
#    for p in img_gray.getdata():
#        print("p: ", p)
    img_resize = img.resize((64, 64), Image.BILINEAR)
    print("img_resize.size: ", img_resize.size)
    img_crop = img.crop((10, 10, 100, 100))
    img_crop.show()
    
    draw = ImageDraw.Draw(img)
    draw.line((0, 0, 60, 60), 'cyan')
    draw.rectangle((10,10,70,90), outline = 'red')
    img.show()
    del draw
        
        

"""而skimage.io读出来的数据是numpy.ndarray格式的, 通道顺序为RGB"""
def skimage_read():
    img_name = "G:\Chinobot\programs\python\image-similarity\images\save_bbox\DJI_0031360_3.jpg"
    img = skimage.io.imread(img_name)
    skimage.io.imshow(img)
    skimage.io.imsave("./io_img.jpg", img)
    pixel = img[20, 10, 2]
    
    print("img.dtype: ", img.dtype)    
    print("type(img): ", type(img))    
    print("img.size: ", img.size)
    print("img.shape: ", img.shape)
    print("img: ", img)
    print("pixel: ", pixel)
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    print("img_width: ", img_width, "img_height: ", img_height)

    img_gray = skimage.color.rgb2gray(img)
    img_resize = skimage.transform.resize(img, (64, 64))
    img_rescale = skimage.transform.rescale(img, [0.5, 0.25])
    img_crop = img[0:64, 0:128, :]
    print("img_gray: ", img_gray)
    print("img_resize.shape: ", img_resize.shape)
    print("img_rescale.shape: ", img_rescale.shape)
    skimage.io.imshow(img_crop)
    
    rr, cc = skimage.draw.line(0, 0, 80, 80)
    print("rr: ", rr)
    print("cc: ", cc)
    img[rr, cc] = (45, 230, 34)
    skimage.io.imshow(img)

    
    
"""使用opencv读取图像，直接返回numpy.ndarray对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255"""
def opencv_read():
    img_name = "G:\Chinobot\programs\python\image-similarity\images\save_bbox\DJI_0031360_3.jpg"
    img = cv2.imread(img_name)
    cv2.imshow("ori", img)    
    cv2.imwrite("./cv2.jpg", img)
    pixel = img[10, 20 ,2]
    
    print("img.dtype: ", img.dtype)
    print("type(img): ", type(img))    
    print("img.size: ", img.size)
    print("img.shape: ", img.shape)
    print("img: ", img)
    print("pixel: ", pixel)
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    print("img_width: ", img_width, "img_height: ", img_height)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img, (64, 64), cv2.INTER_AREA)
    img_crop = img[0:64, 0:128]
    print("img_gray: ", img_gray)
    print("img_resize.shape: ", img_resize.shape)
    cv2.imshow("img_crop", img_crop)

    cv2.line(img, (20, 20), (50, 50), (0, 0, 255), 2)
    cv2.rectangle(img, (45, 45), (88, 88), (0, 255, 0), 2)


    cv2.namedWindow(img_name, 0)
    cv2.resizeWindow(img_name, 640, 480)
    cv2.imshow(img_name, img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()



"""通过上面三个函数可以读取出图片的numpy数据，利用plt模块可以实现显示以及保存"""
def plt_fun():
    image = np.array([0.313660827978, 0.365348418405, 0.423733120134,
                  0.365348418405, 0.439599930621, 0.525083754405,
                  0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)
    plt.imshow(image)
    #plt.imshow(image, interpolation = 'nearest', cmap = 'bone', origin = 'lower')
    """在plt.show()之前调用plt.savefig(), 因为在plt.show()后实际上已经创建了一个新的空白的图片（坐标轴）"""
    plt.savefig('./plt_fig.jpg')
    plt.show()

    

if __name__ == '__main__':
    #PIL_read()
    #skimage_read()
    #opencv_read()
    plt_fun()    
    
    
    
