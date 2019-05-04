# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:33:22 2019

@author: SSG
"""
#################### DEFINE METHODS ####################
import tkinter as tk
from PIL import ImageTk, Image

counter = 0 
image_counter = 0
class_list = [ ("cats",1), ("dogs",2)]
#function to update the counter value
def counter_label(label):
  def count():
    global counter
    counter += 1
    label.config(text=str(counter))
    label.after(1000, count)
  count()

#callback function to reset counter
def reset_counter():
    global counter
    counter = -1
    
def clicked_next(photolabel):
    global logo
    global image_counter
    global class_name
    
    image_counter = image_counter + 1
    img = Image.open("dataset/training_set/" + class_name +"s/" + class_name + "." +str(image_counter) + ".jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    logo = ImageTk.PhotoImage(img)
    #print ("Image height = ",logo.height())
    #print ("Image width = ",logo.width())
    #photoLabel.configure(image = logo)
    photoLabel['image'] = logo

def clicked_prev(photolabel):
    global logo
    global image_counter
    global class_name
    
    if (image_counter < 2) :
        image_counter = 1
    else:
        image_counter = image_counter - 1
    img = Image.open("dataset/training_set/" + class_name +"s/" + class_name + "." + str(image_counter) + ".jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    logo = ImageTk.PhotoImage(img)
    #print ("Image height = ",logo.height())
    #print ("Image width = ",logo.width())
    #photoLabel.configure(image = logo)
    photoLabel['image'] = logo
              
def ShowChoice():
    global class_val
    global class_name
    class_val = v.get()
    if(class_val == 0):
        class_name = "cat"
    else:
        class_name = "dog"    

#################### MAIN ####################

root = tk.Tk()
root.geometry("800x500")
root.title("CNN Demo")
root.configure(bg="white")

#use of Label to show a text message
label1 = tk.Label(root, bg = "white", text = "Hello World")
label1.pack()

label = tk.Label(root, bg = "white", fg="red") #try to use pack in a separate line, and dont do that in the same line with the widget itself
label.pack()
counter_label(label)

#read image using PhotoImage of PIL and use it in a Label
img = Image.open("cat_or_dog_1.jpg")
img = img.resize((250, 250), Image.ANTIALIAS)
logo = ImageTk.PhotoImage(img)
#print (logo.height())
photoLabel = tk.Label(root)
photoLabel.pack(side = tk.RIGHT)
#load_img(img, photoLabel)

button_reset = tk.Button(root, width = 25, text = "reset counter", fg = "blue", bg = "white", command = reset_counter)
button_reset.place(relx = 0.38, rely = 0.1)
#button_reset.pack(side = tk.LEFT)

button_prev = tk.Button(root, width = 25, text = "Prev Sample", fg = "black", bg = "yellow", command = lambda:clicked_prev(photoLabel))
button_prev.place(relx = 0.03, rely = 0.65)
#button_prev.pack(side = tk.LEFT)
button_next = tk.Button(root, width = 25, text = "Next Sample", fg = "black", bg = "yellow", command = lambda:clicked_next(photoLabel))
button_next.place(relx = 0.27, rely = 0.65)
#button_next.pack(side = tk.LEFT)

v = tk.IntVar()
v.set(1)  # initializing the choice, i.e. Python
for val, classes in enumerate(class_list):
    tk.Radiobutton(root, 
                  text=classes,
                  width = 20,
                  indicatoron = 0,
                  variable=v, 
                  command=ShowChoice,
                  value=val).pack(anchor=tk.W)

#button to destroy the gui
button =tk.Button(root, width = 15, text = "Exit Application", fg="black", bg = "red", command = root.destroy)
button.place(rely = 0.9, relx = 0.42)
root.mainloop()
