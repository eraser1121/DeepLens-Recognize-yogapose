import numpy as np
import os

np.random.seed(3)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# 원본 위
optInputPath = '~/yoga/'
# 저장될 위치
optOutputPath = '~/preview/'

# 이미지 크기 조정 비율
optRescale = 1./255
# 이미지 회전
optRotationRange=10
# 이미지 수평 이동
optWidthShiftRange=0.2
# 이미지 수직 이동
optHeightShiftRange=0.2
# 이미지 밀림 강도 
optShearRange=0.5
# 이미지 확대/ 축소 
optZoomRange=[0.9,2.2]
# 이미지 수평 뒤집기 
optHorizontalFlip = True
# 이미지 수직 뒤집기 
optVerticalFlip = False
optFillMode='nearest'
# 이미지당 늘리는 갯수 
optNbrOfIncreasePerPic = 3
# 배치 수 
optNbrOfBatchPerPic = 2

train_datagen = ImageDataGenerator(rescale=optRescale, 
                                   rotation_range=optRotationRange,
                                   width_shift_range=optWidthShiftRange,
                                   height_shift_range=optHeightShiftRange,
                                   shear_range=optShearRange,
                                   zoom_range=optZoomRange,
                                   horizontal_flip=optHorizontalFlip,
                                   vertical_flip=optVerticalFlip,
                                   fill_mode=optFillMode)

def checkFoler(path):
    try:
        if not(os.path.isdir(path)):
            os.makedirs(os.path.join(path))
    except OSError as e:
        if e.errno != errno.EEXIST:                        
            raise            

def increaseImage(path ,folder):
    for index in range(0,optNbrOfIncreasePerPic):                                   
        img = load_img(path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        checkFoler(optOutputPath+folder)               
        print('index : ' + str(index))
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=optOutputPath+folder, save_prefix='tri', save_format='jpg'):
            i += 1
            print(folder + " " + str(i))
            if i >= optNbrOfBatchPerPic: 
                break

def generator(dirName):
    checkFoler(optOutputPath)
    try:
        fileNames = os.listdir(dirName)
        for fileName in fileNames:
            fullFileName = os.path.join(dirName, fileName)
            if os.path.isdir(fullFileName):                
                generator(fullFileName)
            else:
                ext = os.path.splitext(fullFileName)[-1]
                folderName = os.path.splitext(fullFileName)[0].split('/')[-2]
                if(ext == '.jpg'):                    
                    increaseImage(fullFileName, folderName)               
                    
    except PermissionError:
        pass

if __name__ == "__main__":
        generator(optInputPath) 
