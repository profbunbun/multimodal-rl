import xml.dom.minidom
from xml.dom.minidom import parse, parseString
import numpy as np
import pandas as pd

# create a utility clase to get the text from xml nodes


def getMinMax(infile):
    file= infile

    doc = parse(file)

    root = doc.documentElement
          
    edges= root.getElementsByTagName("edge")

    index = 0
    print(doc.nodeName)
    print(doc.firstChild)
    array= np.array([])
    for edge in edges:
           lanes = edge.getElementsByTagName("lane")
           
           for lane in lanes:
                   shape=lane.getAttribute("shape")
                   shape1=shape.split(" ")
                   array= np.append([array],[shape1])
                   
    array2= np.array([])                     
    for element in range(len(array)):
            str1=str(array[element])
            
            str2=str1.split(",")
           
            array2=np.append([array2],[str2])

    array3=np.array([])
    for element in range(len(array2)):
            np.append([array3],[array2[element]])

    df = pd.DataFrame(array2)
    df=pd.to_numeric(df[0],downcast="float")
    max=df.max()
    min=df.min()
    diff=max-min
   
    
    return min,max,diff
        
        
        
        
        
        

    



# file= "nets/3x3.net.xml"

# doc = parse(file)

# root = doc.documentElement
      
# edges= root.getElementsByTagName("edge")

# index = 0
# print(doc.nodeName)
# print(doc.firstChild)
# array= np.array([])
# for edge in edges:
#        lanes = edge.getElementsByTagName("lane")
       
#        for lane in lanes:
#                shape=lane.getAttribute("shape")
#                shape1=shape.split(" ")
#                array= np.append([array],[shape1])
               
# array2= np.array([])                     
# for element in range(len(array)):
#         str1=str(array[element])
        
#         str2=str1.split(",")
       
#         array2=np.append([array2],[str2])

# array3=np.array([])
# for element in range(len(array2)):
#         np.append([array3],[array2[element]])

# df = pd.DataFrame(array2)
# df=pd.to_numeric(df[0],downcast="float")
# print(df.max())
# print(df.min())
# print(df.max()-df.min())



