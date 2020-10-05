# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:44:37 2019

@author: dhy
"""
import NNet_use as nu
import numpy
import theano
import theano.tensor as T


import tkinter as tk
import tkinter.messagebox
import tkinter.ttk
from tkinter import scrolledtext


import cv2 
from PIL import Image, ImageTk
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import datetime
import os

if __name__ == "__main__":


    params0, params1, params2, params3 = nu.load_params()
    nkerns=[5, 10]

    x = T.matrix()    
        
    layer0_input = x.reshape((1, 1, 57, 47))
    

    layer0 = nu.LeNetConvPoolLayer(
        input = layer0_input,
        params = params0,
        image_shape = (1, 1, 57, 47),
        filter_shape = (nkerns[0], 1, 5, 5),
        poolsize = (2, 2)
    )

    layer1 = nu.LeNetConvPoolLayer(
        input = layer0.output,
        params = params1,
        image_shape = (1, nkerns[0], 26, 21),
        filter_shape = (nkerns[1], nkerns[0], 5, 5),
        poolsize = (2, 2)
    )

    layer2_input = layer1.output.flatten(2)
    layer2 = nu.HiddenLayer(
        input = layer2_input,
        params = params2,
        n_in = nkerns[1] * 11 * 8,
        n_out = 2000
        )

    layer3 = nu.Softmax(
            input = layer2.output, 
            params = params3,
            n_in = 2000, 
            n_out = 40
    )   
     
    image_face = Image.open("./disposed.jpg")
    img_arrayed = numpy.array(image_face, dtype='float64') / 256   
    face = theano.shared(img_arrayed, borrow=True)
    
    
    func_believe = theano.function([x],
                           layer3.believe())
    
    func_res = theano.function([x],
                               layer3.result())
    
    

    

    window = tk.Tk()
    window.title("门禁系统")
    window.geometry('400x600')


    def hit_me_open():



        cap = cv2.VideoCapture(0)
        while True:
            ret, showing_image = cap.read(0)
            cv2.imwrite("./now_showing.jpg", showing_image)
            cv2.imshow("show", showing_image)


            now_showing_grayed = cv2.imread("./now_showing.jpg", 0)
            cv2.imwrite("./now_showing_grayed.jpg", now_showing_grayed)

            img = Image.open("./now_showing_grayed.jpg")
            img = img.crop((230, 140, 370, 300))
            img = img.resize((47, 57))
            img.save("./disposed.jpg")
            
            
            
            image_face = Image.open("./disposed.jpg")
            face = numpy.array(image_face, dtype='float64') / 256   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(func_res(face), func_believe(face))
            print()
            if(int(tkenter.get()) == func_res(face) and func_believe(face)>0.3):
                tkinter.messagebox.showinfo('门已开', '欢迎回家')
                fo = open("./records.txt", "a")
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                fo.write("住户：" + tkenter.get() + "\n开门时间：\n" + time + "\n")
                fo.close()
                time = time.replace(":", "m")
                cv2.imwrite("./Photos\\" + time+".jpg", showing_image)
                break
            '''
            if(int(tkenter.get()) == func_res(face) and func_believe(face)>0.6):
                tkinter.messagebox.showinfo('门已开', '欢迎回家')
                fo = open("records.txt", "a")
                fo.write("住户：" + tkenter.get() + "\n开门时间：\n" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
                fo.close()
                
                
                
                break
'''
        cap.release()
        cv2.destroyAllWindows()


    def hit_me_manage():
        if (tkenter_manager.get() == 'dhy'):
            tkinter.messagebox.showinfo("", '登录成功')
            fi = open("./records.txt", "r")
            string = fi.read(1000)
            
            tl = tk.Toplevel()
            tl.title("管理员")
            tl.geometry("800x1000")
            
            tktext = scrolledtext.ScrolledText(tl, width=20, height=10)
            tktext.delete("0.0", "end")
            tktext.insert(0.0, string)
            fi.close()
            tktext.pack()
            
            
            tkenter_start = tk.Entry(tl, width=10)
            tkenter_end = tk.Entry(tl, width=10)
            tkenter_start.pack()
            tkenter_end.pack()
            
            

            

            
            
            def hit_me_query():
                time_start = tkenter_start.get()
                time_end = tkenter_end.get()
                
                tktext.delete(0.0, "end")
                fi = open("./records.txt", "r")
                line1 = ""
                line2 = ""
                line3 = ""

                line1 = fi.readline()
                while(line1):
                    line2 = fi.readline()
                    line3 = fi.readline()
                    if(line3 > time_start and line3 < time_end):
                        tktext.insert("end", line1 + line2 + line3)
                    line1 = fi.readline()
                
                
                fi.close()
                
            button = tk.Button(tl, text="按时间段查询", width=10, height=2, command=hit_me_query)
            button.pack()

            tk.Label(tl, text = "").pack()
            tk.Label(tl, text = "").pack()
            tk.Label(tl, text = "").pack()
            
            tk.Label(tl, text = "输入时间").pack()
            
            
            def hit_me_that():
                time = dateChosen.get()
                img_open = Image.open("./Photos\\" + 
                                      time.replace(":", "m")+".jpg")
                img=ImageTk.PhotoImage(img_open)
                label_img.config(image=img)
                label_img.image=img
                
                
                
            dateChosen = tk.Entry(tl, width=10)
            dateChosen.pack()
            button = tk.Button(tl, text="查看当时影像", width=10, height=2, command=hit_me_that)
            button.pack()


            def hit_me_inf():
                tll = tk.Toplevel()
                tll.title("住户信息")
                tll.geometry("800x500")
            
                def hit_me_yes():
                    number = int(tkenter_no.get())
                    global num
                    num = number
                    
                    
                    img_open = Image.open("./Photos\\1.jpg")
                    img=ImageTk.PhotoImage(img_open)
                    if(number==40):
                        label_img.config(image=img)
                        label_img.image=img
                        name_text.config(text = names[number])
                        name_text.text = names[number]
                    else:
                        image_olive = Image.open("./Photos\\_olivettifaces.jpg")
                        img_arrayed = numpy.array(image_olive, dtype='float64')
                        that = img_arrayed[(number%2)*2850: ((number%2)*2850+285),  int((number/2))*2350: (int((number/2))*2350+235)]
                        img = Image.fromarray(that.astype(numpy.uint8))
                        img.save("./Photos\\temp.jpg")
                        
                        
                        imgg_open = Image.open("./Photos\\temp.jpg")
                        imgg=ImageTk.PhotoImage(imgg_open)
                        label_img.config(image=imgg)
                        label_img.image=imgg
                        name_text.config(text = names[number])
                        name_text.text = names[number]
                    


                    
                tkenter_no = tk.Entry(tll, width=10)
                tkenter_no.pack()
                button_yes = tk.Button(tll, text="确定", width=10, height=2, command=hit_me_yes)
                button_yes.pack()
                name_text = tk.Label(tll, text = "")
                name_text.pack()
                label_img = tk.Label(tll)
                label_img.pack()
                
                
                def hit_me_rec():
                    tlll = tk.Toplevel()
                    tlll.title("修改姓名信息")
                    tlll.geometry("500x300")
                    tk.Label(tlll, text = "输入新姓名").pack()
                    new_name = tk.Entry(tlll)
                    new_name.pack()
                    
                    def hit_me_ok():
                        names[num] = new_name.get()
                        
                        
                    button_ok = tk.Button(tlll, text="确定", width=10, height=2, command=hit_me_ok)
                    button_ok.pack()
                    
                    
                button_rec = tk.Button(tll, text="修改用户姓名", width=10, height=2, command=hit_me_rec)
                button_rec.pack()
                

            
            
            
            button_inf = tk.Button(tl, text="管理用户信息", width=10, height=2, command=hit_me_inf)
            button_inf.pack()
            
            label_img = tk.Label(tl)
            label_img.pack()

            
            

            

        else :
            tkinter.messagebox.showinfo("", '登录失败')



    b1 = tk.Button(window, text="开门", width=10, height=2, command=hit_me_open)
    b2 = tk.Button(window, text="管理员登录", width=10, height=2, command=hit_me_manage)

    tkenter = tk.Entry(window, width=10)



    tkenter_manager = tk.Entry(window, width=10)




    tk.Label(window, text = "    ", width=10, height=1).pack()

    b1.pack()
    tk.Label(window, text = "门牌号", width=10, height=2).pack()
    tkenter.pack()
    tk.Label(window, text = "    ", width=10, height=2).pack()
    tk.Label(window, text = "    ", width=10, height=2).pack()
    tk.Label(window, text = "    ", width=10, height=2).pack()




    b2.pack()
    tk.Label(window, text="管理员密码", width=10, height=2).pack()
    tkenter_manager.pack()
    tk.Label(window, text = "    ", width=10, height=2).pack()


    '''
    canvas = tk.Canvas(window, width=1000, height=600)
    myImage = ImageTk.PhotoImage(img)
    canvas.create_image(350, 50, anchor="nw", image=myImage)
    canvas.place(x=350, y=50)
    '''

    img_open = Image.open("./Photos\\1.jpg")
    img=ImageTk.PhotoImage(img_open)
    
    
    names = []
    for i in range(41):
        names.append("")
    names[40] = "董泓源"
    for i in range(40):
        names[i] = str(i)
    num = 0


    window.mainloop()
    

    os.system("pause")


