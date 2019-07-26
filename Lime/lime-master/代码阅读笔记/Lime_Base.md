# Lime_Base

-

<b>Class</b> LimeBase:

_@Constructor_  
1.\_\_init\_\_(kernel\_fn,verbose=false,random\_state=None)  

_@StaticMethod_  
2.generate\_lars\_path(weighted\_data, weighted\_labels)  
&ensp;&ensp;&ensp;&ensp;通过已赋予权值的数据和标签，训练出lars_path并且返回alpha和coefs

_@MemberFunction_  
3.forward_selection(self, data, labels, weights, num\_features)  
&ensp;&ensp;&ensp;&ensp;前向选择算法  

_@MemberFunction_  
4.feature\_selection(self, data, labels, weights, num\_features, method)  
&ensp;&ensp;&ensp;&ensp;特征选择的过程(前向选择,最高权重,lasso法,自动选择)  

_@MemberFunction_  
5.explain\_instance\_with\_data(self,neighborhood\_data,neighborhood\_labels,  distances,label,num\_features,feature\_selection='auto',model\_regressor=None)  
&ensp;&ensp;&ensp;&ensp;传入扰动后的数据,标签和距离。返回最后的explanation。  
&ensp;&ensp;&ensp;&ensp;Return:(intercept, exp, score, local_pred)
