#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, cv2, pdb
import numpy as np


# In[2]:


# import os, cv2, pdb
# import numpy as np
# temp_list=[[20,60,200,300],[10,20,400,300]]
# temp_list=np.array(temp_list)
# count=0
# bbox_temp=[]
# for i in range(4):
#     if count<2:
#         if temp_list[0][count]>=temp_list[1][count]:
#             bbox_temp.append(temp_list[1][count])
#         else:
#             bbox_temp.append(temp_list[0][count])
#     else:
#         if temp_list[0][count]<=temp_list[1][count]:
#             bbox_temp.append(temp_list[1][count])
#         else:
#             bbox_temp.append(temp_list[0][count])
#     count+=1

# print(bbox_temp)


# In[3]:


import xml.etree.ElementTree as ET
def GetAnnotBoxLoc(AnotPath,image):#AnotPath VOC標註文件路徑
    tree = ET.ElementTree(file=AnotPath)  #打開文件，解析成一棵樹型結構
    root = tree.getroot()#獲取樹型結構的根
    ObjectSet=root.findall('object')#找到文件中所有含有object關鍵字的地方，這些地方含有標註目標
    ObjBndBoxSet={} #以目標類別爲關鍵字，目標框爲值組成的字典結構
    bbox=[]
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因爲程序是按0作爲起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc=[x1,y1,x2,y2]
        bbox_singel=[x1,y1,x2,y2,0]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return image


# In[ ]:


import cv2
import numpy as np
f =open("/ssd3/u1/NBI_NET/EfficientDet-master/datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt","r")
# test context
# 2144 -> image_name 
# 4455 -> image_name
# 1473 -> image_name
# ....
image_txt=f.readlines()
# print(image_txt)
f.close()
f=open("/ssd3/u1/NBI_NET/EfficientDet-master/bbox.txt","r")
# bbox context
# 114.95752882957458,111.8162602186203,435.5493502020836,480.0
# 94.8961410522461,202.71891021728516,405.2527046203613,480.0
# 104.09564208984375,142.97987174987793,424.7352819442749,480.0
# 22.32706391811371,165.31551712751389,323.2886560857296,480.0
# xmin ymin xmax ymax↑

bbox_path=f.readlines()
bbox_path[-1].replace('\n','')
f.close
bbox_list=[]
image_ann="/ssd3/u1/NBI_NET/EfficientDet-master/datasets/VOCdevkit/VOC2007/Annotations2/"
image_path="/ssd3/u1/NBI_NET/EfficientDet-master/datasets/VOCdevkit/VOC2007/JPEGImages/"
imageSeg_path="/ssd3/u1/NBI_NET/EfficientDet-master/datasets/VOCdevkit/VOC2007/SegmentationClass/"


count=0
# f=open("/ssd3/u1/NBI_NET/NBI/IMG/test20_L2_crop.txt",'w')
f=open("/ssd3/u1/NBI_NET/NBI/IMG/test20_L2_crop_ann2.txt",'w')
for i in range(len(bbox_path)):
    bbox_temp=[]
    count+=1
    bbox_path[i].replace('\n','')
    xmin,ymin,xmax,ymax=bbox_path[i].split(',',4)
    xmin,ymin,xmax,ymax=float(xmin), float(ymin), float(xmax), float(ymax)
    image=cv2.imread(image_path+(image_txt[i].replace('\n',''))+'.png')
    
#   ↓將annotation的座標可視化-------------------------------------------------------------
#     image=GetAnnotBoxLoc(image_ann+image_txt[i-1].replace('\n','')+'.xml',image)
#     cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    
#   -------------------------------------------------------------------------------------
#     image_seg=cv2.imread(imageSeg_path+(image_txt[i].replace('\n',''))+'.png')
#     image_seg=cv2.resize(image_seg,(480,480),interpolation=cv2.INTER_CUBIC)
    crop_img=image[int(ymin):int(ymax),int(xmin):int(xmax)]
    crop_img=cv2.cv2.resize(crop_img,(480,480),interpolation=cv2.INTER_CUBIC)
    
#     crop_seg=image_seg[int(ymin):int(ymax),int(xmin):int(xmax)]
#     crop_seg=cv2.cv2.resize(crop_seg,(480,480),interpolation=cv2.INTER_CUBIC)
    
    htitch=np.hstack((image,image_seg))
#     cv2.imshow("my image",image)
    cv2.imshow("my image",htitch)


    crop_seg_path="/ssd3/u1/NBI_NET/NBI/IMG/TestData_13_images/final_20_segmap_crop_ann2/%s.png"%(image_txt[i].replace('\n',''))
    crop_img_path="/ssd3/u1/NBI_NET/NBI/IMG/TestData_13_images/final_20_images_crop_ann2/%s.png"%(image_txt[i].replace('\n',''))
    f.write("TestData_13_images/final_20_images_crop_ann2/"+image_txt[i-1].replace('\n','')+".png TestData_13_images/final_20_segmap_crop_ann2/"+image_txt[i].replace('\n','')+".png 1 1\n")
    print(crop_seg_path)
    cv2.imwrite(crop_seg_path,crop_seg)
    cv2.imwrite(crop_img_path,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
f.write("TestData_13_images/final_20_images_crop_ann2/t2_104.png TestData_13_images/final_20_segmap_crop_ann2/n1.png 0 1\n")
f.write("TestData_13_images/final_20_images_crop_ann2/t2_105.png TestData_13_images/final_20_segmap_crop_ann2/n1.png 0 1\n")
f.write("TestData_13_images/final_20_images_crop_ann2/t2_11.png TestData_13_images/final_20_segmap_crop_ann2/n1.png 0 1\n")
f.write("TestData_13_images/final_20_images_crop_ann2/t2_164.png TestData_13_images/final_20_segmap_crop_ann2/n1.png 0 1\n")
f.write("TestData_13_images/final_20_images_crop_ann2/t2_570.png TestData_13_images/final_20_segmap_crop_ann2/n1.png 0 1\n")
f.write("TestData_13_images/final_20_images_crop_ann2/t2_773.png TestData_13_images/final_20_segmap_crop_ann2/n1.png 0 1\n")
f.close()
print("count",count)


# In[ ]:


# b=[]
# c=[]

# z="[/ssd3/u1/NBI_NET/tensorflow-yolov3/checkpoint/test.png]"
# zz=z[1:-1]
# print(z)

# b.append("/ssd3/u1/NBI_NET/tensorflow-yolov3/checkpoint/test.png")
# b.append("/ssd3/u1/NBI_NET/tensorflow-yolov3/checkpoint/test2.png")
# c.append('0')
# c.append('1')
# c=np.array(c) # 200,1
# b=np.array(b) # 200,1
# # augmentation
# # b

# all_temp=[]
# for i in range(len(b)):
#     temp=[]
#     temp.append(b[i])
#     temp.append(c[i])
#     all_temp.append(temp)
    

# # print(all_temp)
# cc=np.array(all_temp)
# print(cc)
# print(cc[0])
# cc=cc.flatten()
# # print(cc.shape)
# count=0


# In[ ]:


##get object annotation bndbox loc start 
# import xml.etree.ElementTree as ET
# def GetAnnotBoxLoc(AnotPath):#AnotPath VOC標註文件路徑
#     tree = ET.ElementTree(file=AnotPath)  #打開文件，解析成一棵樹型結構
#     root = tree.getroot()#獲取樹型結構的根
#     ObjectSet=root.findall('object')#找到文件中所有含有object關鍵字的地方，這些地方含有標註目標
#     ObjBndBoxSet={} #以目標類別爲關鍵字，目標框爲值組成的字典結構
#     bbox=[]
#     for Object in ObjectSet:
#         ObjName=Object.find('name').text
#         BndBox=Object.find('bndbox')
#         x1 = int(BndBox.find('xmin').text)#-1 #-1是因爲程序是按0作爲起始位置的
#         y1 = int(BndBox.find('ymin').text)#-1
#         x2 = int(BndBox.find('xmax').text)#-1
#         y2 = int(BndBox.find('ymax').text)#-1
#         BndBoxLoc=[x1,y1,x2,y2]
#         bbox_singel=[x1,y1,x2,y2,0]
# #         if ObjName in ObjBndBoxSet:
# #             ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典結構中含有這個類別了，那麼這個目標框要追加到其值的末尾      
# #         else:
# #             ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典結構中沒有這個類別，那麼這個目標框就直接賦值給其值吧
#         bbox.append(bbox_singel)
#     print(bbox)
#     return bbox
##get object annotation bndbox loc end

# if __name__== '__main__':
#     datatype="train"
#     data_path="/ssd3/u1/NBI_NET/VOC_NBI/"+datatype+"/VOCdevkit/VOC2007/JPEGImages"
#     anno_path="/ssd3/u1/NBI_NET/VOC_NBI/"+datatype+"/VOCdevkit/VOC2007/Annotations"
#     dp=os.listdir(data_path)#113
#     ap=os.listdir(data_path)#113
#     txt_path="/ssd3/u1/NBI_NET/tensorflow-yolov3/data/dataset/voc_train.txt"
#     data=open(txt_path,'w')
    
#     for i in dp:
#         img_path=[0]
#         img_path = data_path+"/"+i
#         bbox=GetAnnotBoxLoc((anno_path + "/")+ i[0:-4]+".xml")
# #         print(img_path)
#         bbox_len=len(bbox)
#         data.write(img_path+" ")
#         for j in range(len(bbox)):
#             temp_bbox=str(bbox[j])
# #             temp_bbox.
#             print(temp_bbox)
#             data.write(temp_bbox[1:-1].replace(', ',',')+" ")
        
#         data.write("\n")
#     data.close()
    
    

    
    
#     get_img_bbox.appen
#     single_label.append(get_img_bbox)


# In[ ]:


# path="/ssd3/u1/NBI_NET/VOC_NBI/train/VOCdevkit/VOC2007/JPEGImages/"
# data=open("/ssd3/u1/NBI_NET/VOC_NBI/train/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt","w")

# allFileListXML=os.listdir(path)
# print(allFileListXML)

# for i in allFileListXML:
#     data.write(i[0:-4]+"\n")
# data.close()


# In[ ]:


# import os, cv2, pdb
# import numpy as np
# pth= '/ssd3/u1/NBI_NET/mmdetection/data/VOCdevkit/VOC2007/JPEGImages/'
# dx=os.listdir(pth)
# len1 = len(dx)
# R,G,B=0,0,0
# for f in dx:
#     fn=pth+f
#     im1=cv2.imread(fn)
#     im1 = np.array(im1, np.float)
#     im1 = np.mean(np.mean(im1, 0),0)
#     R,G,B = R+im1[2], G+im1[1], B+im1[0]

# R,G,B =R/len1,G/len1,B/len1
# print(R,G,B)
    


# In[ ]:


# data_val='train'
# dataset='cifar10'  
# #     通過%s更改資料格式或是來源 
# source_path="../cifar2png/%sclass/%s/" % (dataset,data_val) 
# target_path="../cifar2png/%s/" % (dataset)
# class_name=[]
# id=0
# images=0

# for classes in os.listdir(source_path):
#     class_name.append([classes])
# #     print(str(classes))
#     for image in os.listdir(source_path+'/'+str(classes)):
#         source_path_image=source_path+str(classes)+'/'+str(image)
#  　　#路徑的最後一定要是檔名 要不然會出現權限不足的錯誤
#         print(source_path_image)
#         shutil.copy(source_path_image, target_path)


# In[ ]:


# trainset=list([('./data/dogcat_2/cat/cat.12484.jpg', 0), ('./data/dogcat_2/cat/cat.12485.jpg', 0), 
#          ('./data/dogcat_2/cat/cat.12486.jpg', 0), ('./data/dogcat_2/cat/cat.12487.jpg', 0), 
#          ('./data/dogcat_2/dog/dog.12496.jpg', 1), ('./data/dogcat_2/dog/dog.12497.jpg', 1), 
#          ('./data/dogcat_2/dog/dog.12498.jpg', 1), ('./data/dogcat_2/dog/dog.12499.jpg', 1)])


# print(np.array(a).shape)
# print(a[0][:])
# print(a[1][:])
# print(a[0][0])
# print(a[0][1])
# # print(len(a))
# real=[]
# false=[]

# for i in range(len(trainset)):
#     if (a[i][1])==0:
#         real.append(a[i][:])
#     else:
#         false.append(a[i][:])

# print(real)
# print(false)


# In[ ]:




