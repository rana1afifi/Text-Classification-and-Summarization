import Summarization as sm
import Classification as cl
import os
import sys

directory = os.getcwd()

module=int(input("Enter 1 for Text Classification and 2 for Text Summarization"))
if module==1:
    inputPath= input("Please copy the file you want to classify to the Test Classify Folder and enter its name ")
    ##cl.classification_pipeline()
    cl.classify(directory+"/Test Classify/"+inputPath)
elif module==2:
    output=input("Enter output filename: ")
    outputFile = open(output, 'w+')
    compression = input("Enter Compression value [from 0 to 1: the lower the value, the shorter the summary]:")
    inputPath = input("Please copy the file you want to summarize to the Test Summarization Folder and enter its name ")
    output = sm.summarize(float(eval(compression)), directory+"/Test Summarization/"+inputPath)
    outputFile.write(output)

else:
    print("Incorrect choice, program will terminate")


