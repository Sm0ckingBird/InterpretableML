# Lime_Text 

-

<b>Class</b> TextDomainMapper:  

&ensp;&ensp;&ensp;&ensp;用于将抽象的feature\_id转换为对应的单词

-

<b>Class</b> IndexedString:  
&ensp;&ensp;&ensp;&ensp;对一个原始的句子进行分词,建立单词表List,位置List,以方便LimeTextExplainer产生扰动的数据。

-


<b>Class</b> LimeTextExplainer:   
&ensp;&ensp;&ensp;&ensp;同样主要是产生扰动数据,然后传入Lime_Base中,返回得到Explanation。

