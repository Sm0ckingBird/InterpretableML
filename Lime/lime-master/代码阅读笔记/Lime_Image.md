# Lime_Image
_只能用于处理RGB格式的图片_

-

<b>Class</b> ImageExplanation:  
_@Constructor_  
1.\_\_init\_\_(...)  
&ensp;&ensp;&ensp;&ensp;初始化属性Image,segments,创建intercept,local\_exp,local\_pred属性

_@MemberFunction_   
2.get\_image\_and\_mask(...)  
&ensp;&ensp;&ensp;&ensp;Return:(image, mask), where image is a 3d numpy array and mask is a 2d numpy array.

&ensp;&ensp;&ensp;&ensp;  这个方法主要用于将图片的解释可视化,画出segments的边界,和为不同的权值赋不同的颜色。




-
<b>Class</b> LimeImageExplainer:  
_@Constructor_  
1.\_\_init\_\_(...)  
&ensp;&ensp;&ensp;&ensp;继承自Lime\_Base,初始化参数(kernel\_fn,feature\_selection)

_@MemberFunction_  
2.explain\_instance(...)  
&ensp;&ensp;&ensp;&ensp;-如果图像为灰度图像则转换成RGB图像  
&ensp;&ensp;&ensp;&ensp;-利用SegmentationAlgorithm对图像进行分割,产生超像素   
&ensp;&ensp;&ensp;&ensp;-产生扰动数据  
&ensp;&ensp;&ensp;&ensp;-产生距离矩阵  
&ensp;&ensp;&ensp;&ensp;-传入参数给ImageExplanation,返回一个Explanation对象

_@MemberFunction_  
3.data\_labels(...)  
&ensp;&ensp;&ensp;&ensp;产生当前要解释样本的扰动数据  
&ensp;&ensp;&ensp;&ensp;返回:  
&ensp;&ensp;&ensp;&ensp;元组 (data, labels):   
&ensp;&ensp;&ensp;&ensp;data: dense num\_samples * num\_superpixels  
&ensp;&ensp;&ensp;&ensp;labels: prediction probabilities matrix