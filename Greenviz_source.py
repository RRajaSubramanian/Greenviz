# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:15:48 2020

@author: shiri
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from tkinter import filedialog
from tkinter import Tk
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import tkinter
from tkinter import*
from matplotlib.figure import Figure
import webbrowser
from PIL import ImageTk , Image
def panfiles():
    global reslist1
    global inde_var,reslist,de_var
    global file,df,col1,df2,df3
    reslist=[]
    reslist1=[]
    file=filedialog.askopenfilename()
    suc_label=Label(frame,text="Succesful!")
    print(file)
    suc_label.grid(row=3)
    try:
        df2=pd.read_csv(file)
        df=df2
        df3=df2
        print(df2)
    except FileNotFoundError:
        messagebox.showerror("Error","Please select a file")
    else:
        col=list(df.columns)
        for i in range(len(col)):
            df.rename(columns={col[i]:str(col[i]).lower()},inplace=True)
        col1=list(df.columns)
        inde_var.set(col1)
        b=Button(frame01,text="*Select",command=selection,borderwidth="3")
        b.grid(row=9,column=1)
    
    
def selection():
    inde_var.set(col1)
    reslist1=[]
    flag=0
    seleccion = inde_enter.curselection()
    for i in seleccion:
        entrada = inde_enter.get(i)
        reslist.append(entrada)
    for i in col1:
        if i not in reslist:    
            reslist1.append(i)
    de_enter=OptionMenu(frame01 ,de_var,*reslist1)
    de_enter.grid(row=10, column=1,padx=2)
    de_b=Button(frame01,text=" *Select",command=pred,borderwidth="3")
    de_b.grid(row=11,column=1)

def pred():
    global inp,oup,cate,cont
    oup=[de_var.get()]
    cont=[]
    cate=[]
    inp=reslist
    global inp1,num,i
    i=0
    num=len(inp)
    sel=Label(frame01,text="select type of the feature")
    sel.grid(row=12,column=0)
    select_fea()
    print(cate)
    print(cont)
def select_fea():
    global num,i,val,v,cate,cont

    v=IntVar()
    lb=Label(frame01,text=inp[i])
    lb.grid(row=13,column=0)
    rb=Radiobutton(frame01,text="continuous",variable=v,value=1)
    rb.grid(row=13,column=1)
    rb=Radiobutton(frame01,text="categorical",variable=v,value=2)
    rb.grid(row=14,column=1)
    nex=Button(frame01,text="next",command=nextone)
    nex.grid(row=15,column=1)
def nextone():
    global valu,i,num,v,cate,cont,inp
    valu=v.get()
    if(valu!=1):
        cate.append(inp[i])
    else:
        cont.append(inp[i])
    
    print(valu)
    i+=1
    if(i<num):
        select_fea()
    
    
    
   # frame03=LabelFrame(main_frame,borderwidth=6 ,text="Greenviz",font="25", padx=10 , pady=5,width=1000,height=1000)
   # frame.grid(padx=10,pady=10)
    
def sub():
    
    global clicked,alg
    global tarin_p,frame2,xl,yl
    global test_p,x,y,train_percent,test_percent
    try:
        x=inp
        xl=x
        y=oup
        yl=y
    except NameError:
        messagebox.showerror("Error","please select the features and click the Select button")
    else:
        try:
            train_percent=float(train_p.get())
            test_percent=float(test_p.get())
        except ValueError:
            messagebox.showerror("Error","please enter valid train and test percent")
        else:
            alg=clicked.get()
            dest()
            bg_image=ImageTk.PhotoImage(Image.open("bg1.jpg"))
            bg_label= Label(root,image=bg_image)
            bg_label.image=bg_image
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            frame2=LabelFrame(root ,borderwidth=6 ,text="Model Building",font="25", padx=10, pady=10,width=1000,height=1000)
            frame2.grid(padx=15,pady=15)
            s_label=Label(frame2,text=" * Your response has been succesfully Recorded ",font="5")
            s_label.grid(row=1,padx=80,pady=8)
            algo=Label(frame2,text=" * "+alg+" Model",font="5")
            algo.grid(row=2,pady=10)
            tr_button=Button(frame2,text=" * Train",command=train_d,width="15",font="5",borderwidth="4")
            tr_button.grid(row=3)
            tr_label=Label(frame2,text="Click here to train the model")
            tr_label.grid(row=4)
            back_b=Button(frame2,text="Back",width="20",font="20",borderwidth="3",command=main)
            back_b.grid(row=0,column=4)
def train_d():  
    global cate_data
    cate_data=df[cate]
    global xtrain,ytrain,ytest,xtest
    try:
        from sklearn.model_selection import train_test_split
        xtrain,xtest,ytrain,ytest=train_test_split(df[x],df[y],train_size=train_percent/100,test_size=test_percent/100)
    except:
        messagebox.showerror("Error","Please enter valid training and test percentage")
        main()
    else:
        
        if(alg=="Linear Regression"):

            linreg()
        elif(alg=="Logistic Regression"):
            print(alg)
        
            logreg()
        else:
            print(alg)
            print("#################################################")
            dtc()
def lin():
    global alg
    alg="Linear Regression"
    clicked.set("Linear Regression")
def log():
    global alg
    clicked.set("Logistic Regression")
    alg="Logistic Regression"
def dtc():
    global linr
    k=1
    try:
        global xtrain,ytrain,xtest,ytest,labelenc,cont,cate
        from sklearn.tree import DecisionTreeClassifier
        linr=DecisionTreeClassifier(max_depth=4)
        labelenc=LabelEncoder()
        print(cate)
        for i in cate:
            print(i)
            labelenc.fit(df[i])
            df[i]=labelenc.transform(df[i])
            print(df[i])
        x=cont
        cont=tuple(cont)
        x.extend(cate)
        cont=list(cont)
        from sklearn.model_selection import train_test_split
        xtrain,xtest,ytrain,ytest=train_test_split(df[x],df[y],train_size=train_percent/100,test_size=test_percent/100)
        linr.fit(xtrain,ytrain)
    except ValueError:
        messagebox.showerror("Error","your selected dependent and independent are producing value errors for training the model \n go back and try another algorithm")
        frame2.destroy()
        main()
    else:
        stat2()
def stat2():
    global y1
    tl=Label(frame2,text=" * Training statistics:: ",font="5")
    tl.grid(row=1,pady=10,column=1)
    from sklearn import metrics
    
    st1=Label(frame2,text=" * No of training samples:: "+"  * "+str(len(xtrain)))
    st1.grid(row=6,column=0)
    st3=Label(frame2,text=" * total no of smaples::"+ "  * "+str(len(xtrain)+len(xtest)))
    st3.grid(row=5,column=1)
    st5=Label(frame2,text=" * training accuracy::"+"  * "+str(metrics.accuracy_score(ytrain,linr.predict(xtrain))))
    st5.grid(row=6,column=1)
    st7=Label(frame2,text="Confusion Matrix",font="5")
    st7.grid(row=7,column=0)
    global cnf_matrix
    cnf_matrix=metrics.confusion_matrix(ytrain,linr.predict(xtrain))
    cnf_button=Button(frame2,text="Click here to view the confusion matrix",command=view_conf,borderwidth="3")
    cnf_button.grid(row=8)
        
    st8=Label(frame2,text=" * mean of independent and dependent variables",font="5")
    st8.grid(row=8,column=1,pady=5)
    y2=[]
    for i in range(len(x)):
        y2.append("*"+x[i]+"  *"+str(df[x[i]].mean()))
    for i in range(len(y2)):
        mean=Label(frame2,text=y2[i])
        mean.grid(row=i+10,column=1)
    index=i+11
    ypr=linr.predict(xtrain)
    print(ytrain,ypr)
    f_sco=metrics.precision_recall_fscore_support(ytrain,linr.predict(xtrain),average="weighted")
    pres=Label(frame2,text="* precison ::  "+" * "+str(f_sco[0]))
    pres.grid(row=2,column=1)
    recall=Label(frame2,text=" * recall  ::"+" * "+str(f_sco[1]))
    recall.grid(row=3,column=1)
    fscore=Label(frame2,text="* Fscore ::"+" * "+str(f_sco[2]))
    fscore.grid(row=4,column=1)
    global unique
    global uv
    uniq=df[y].nunique()
    uv=list(uniq)
    dtc_graph=Button(frame2,text="Click here to view Decision tree",command=view_tree,borderwidth="3")
    dtc_graph.grid(row=index+1,column=0)
    test_b=Button(frame2,text="Test and visualisation",width="20",font="5",padx=80,borderwidth="5",command=test_but)
    test_b.grid(row=index+2,pady=20,padx=100)
    test_l=Label(frame2,text="Click here to validate/test the data")
    test_l.grid(row=index+3)
    global clf
    clf=linr.fit(xtrain,ytrain)
    
def view_tree():
    import pydotplus
    global clf
    import matplotlib.pyplot as plt
    
    from sklearn.tree import export_graphviz
    import graphviz

    
    
    dot_file=export_graphviz(linr,feature_names=x)
    graph=graphviz.Source(dot_file)
    graph.render(filename="tree2",format="png",cleanup=True)
    top=Toplevel()
    dl=Label(top,text="Decision tree")
    img = Image.open("tree2.png")  # PIL solution
    img = img.resize((1500, 500), Image.ANTIALIAS) #The (250, 250) is (height, width)
    img = ImageTk.PhotoImage(img)
    dtc_label=Label(top, image=img )
    dtc_label.image=img
    dtc_label.grid(row=1,column=0)
    
    
    
    
    
        
def linreg():
    global linr
    try:
        
        global xtrain,ytrain,xtest,ytest
        linr=LinearRegression()
        linr.fit(xtrain,ytrain)
    except ValueError:
        messagebox.showerror("Error","your selected dependent and independent are producing value errors for training the model \n go back and try another algorithm")
        frame2.destroy()
        main()
    else:
        stat()
def logreg():
    global linr
    try:
        global ytrain,ytest,xtrain,xtest
        y1=np.array(df[y])
        from sklearn.model_selection import train_test_split
        xtrain,xtest,ytrain,ytest=train_test_split(df[x],y1,train_size=train_percent/100,test_size=test_percent/100)
        
        linr=LogisticRegression(solver="lbfgs")
        xtrain,xtest=np.array(xtrain),np.array(xtest)
        linr.fit(xtrain,ytrain)
    except ValueError:
        messagebox.showerror("Error","your selected dependent and independent are producing value errors for training the model \n go back and try another algorithm")
        frame2.destroy()
        main()
    else:
        stat1()
def test_but():
    if(alg=="Linear Regression"):
        test_but_lin()
    elif(alg=="Logistic Regression"):
        test_but_log()
    else:
        test_but_dtc()
def test_but_dtc():
    frame2.destroy()
    global frame3
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import plot_precision_recall_curve
    frame3=LabelFrame(root ,borderwidth=6 ,text="Test and visualisation",font="30", padx=30 , pady=30,width=1000,height=1000)
    frame3.grid(padx=200,pady=100)
    tl=Label(frame3,text=" * Testiing Statistics:: ",font="15")
    tl.grid(row=1,pady=10,column=0)
    from sklearn .metrics import r2_score
    st1=Label(frame3,text=" * No of Testing Samples:: "+"  * "+str(len(xtest)))
    st1.grid(row=2,column=0)
    from sklearn import metrics
    st5=Label(frame3,text=" * Testing Accuracy::"+"  * "+str(metrics.accuracy_score(ytest,linr.predict(xtest))))
    st5.grid(row=3,column=0)
    st7=Label(frame3,text="Confusion Matrix",font="5")
    st7.grid(row=4,column=0)
    global cnf_matrix
    
    cnf_matrix=metrics.confusion_matrix(ytest,linr.predict(xtest))
    print(cnf_matrix)
    cnf_button=Button(frame3,text="Click here to view confusion matrix",command=view_conf,borderwidth="3")
    cnf_button.grid(row=5)

    index=6
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plots=Label(frame3,text=" * Select your plot ",pady=20,font="15")
    plots.grid(row=index,column=0,pady=10)
    global charts,plot_var,df1
    charts=["line plot","Bar","histplot","precision_recall"]
    back_2=Button(frame3,text="Back",font="5",width="10",borderwidth="3",command=sub)
    back_2.grid(row=0,column=4)
    option_b=Label(frame3,text="Enter your predictors",font="5")
    option_b.grid(row=4,column=2)
    global cont_fea
    cont_fea=[]
    prediction_button2()
def prediction_button2():
    global ind
    ind=0
    global x_val
    x_val=StringVar()
    global flag
    flag=1
    global fea
    fea=[]
    fun=prediction10()
def prediction10():
    global ind,xl,eg,lg
    print(cont)
    print(cate)
    lg=Label(frame3,text=cont[ind],font="5")
    lg.grid(row=5,column=1)
    global eg
    eg=Entry(frame3,textvariable=x_val)
    eg.grid(row=5,column=2)
        
    b=Button(frame3,text="click",command=dirs01)
    b.grid(row=5,column=3)
def dirs01():
    global ind
    global fea
    fea.append(float(x_val.get()))
    print(fea)
    if(ind<len(cont)-1):
        print(ind)
        flag=1
        eg.destroy()
        x_val.set("")
        
        ind+=1
        prediction10()
    else:
        global subm
        if(len(cate)>0):
            eg.destroy()
            global lg
            lg.destroy()
            ind=0
            prediction01()
    
def prediction01():
    global eg,lb
    global ind,sel_var,df3
    lb=Label(frame3,text=cate[ind],font="5")
    lb.grid(row=5,column=1)
    global cate_data
    print(cate_data)
    sel_var=StringVar()
    l=cate_data[cate[ind]].unique()
    print(cate_data)
    eg=OptionMenu(frame3,sel_var,*l)
    eg.grid(row=5,column=2)
    b=Button(frame3,text="click",command=dirs02)
    b.grid(row=5,column=3)
def dirs02():
    global ind
    global cont_fea
    cont_fea.append(sel_var.get())
    print(cont_fea)
    if(ind<len(cate)-1):
        print(ind)
        flag=1
        global lb
        eg.destroy()
        lb.destroy()
        sel_var.set("")
        ind+=1
        prediction10()
    else:
        global subm
        subm=Button(frame3,text="View Result",command=view_res01)
        subm.grid(row=6,column=2)
    
    
    
    #eg=OptionMenu(frame3,sel_var,)
    
def view_res01():
    r=[]
    global subm,fea,cont_fea,enc,subm,eg
    r=fea
    enc=[]
    for i in range(len(cont_fea)):
        labelenc=LabelEncoder()
        labelenc.fit([cont_fea[i]])
        enc.extend(labelenc.transform([cont_fea[i]]))
    r.extend(enc)
    r=[r]
    
    re=linr.predict(r)
    global result_lab,bac
    print(re)
    result_lab=Label(frame3,text="Result is "+str(re[0]),font="5")
    subm.destroy()
    result_lab.grid(row=6,column=2)
    bac=Button(frame3,text="Clear",command=clear01)
    bac.grid(row=6,column=3)
def clear01():
    global result_lab,bac,fea,eg,ind,cont_fea
    cont_fea=[]
    fea=[]
    ind=0
    cont_fea=[]
    bac.destroy()
    eg.destroy()
    result_lab.destroy()
    print("###################################################")
    prediction_button2()
    
    
def test_but_log():
    frame2.destroy()
    global frame3
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import plot_precision_recall_curve
    frame3=LabelFrame(root ,borderwidth=6 ,text="Test and visualisation",font="30", padx=30 , pady=30,width=1000,height=1000)
    frame3.grid(padx=200,pady=100)
    tl=Label(frame3,text=" * Testiing Statistics:: ",font="15")
    tl.grid(row=1,pady=10,column=0)
    from sklearn .metrics import r2_score
    st1=Label(frame3,text=" * No of Testing Samples:: "+"  * "+str(len(xtest)))
    st1.grid(row=2,column=0)
    from sklearn import metrics
    st5=Label(frame3,text=" * Testing Accuracy::"+"  * "+str(metrics.accuracy_score(ytest,linr.predict(xtest))))
    st5.grid(row=3,column=0)
    st7=Label(frame3,text="Confusion Matrix",font="5")
    st7.grid(row=4,column=0)
    global cnf_matrix
    cnf_matrix=metrics.confusion_matrix(ytest,linr.predict(xtest))
    print(cnf_matrix)
    cnf_button=Button(frame3,text="Click here to view confusion matrix",command=view_conf,borderwidth="3")
    cnf_button.grid(row=5)
    c=[]
    index=6
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plots=Label(frame3,text=" * Select your plot ",pady=20,font="15")
    plots.grid(row=index,column=0,pady=10)
    global charts,plot_var,df1
    charts=["line plot","Bar","histplot","precision_recall"]
    if(uv[0]>2):
        charts.remove("precision_recall")
    plot_var=StringVar()
  #  plot_var().set(charts)
    plt_box=OptionMenu(frame3,plot_var,*charts)
    plt_box.grid(row=index+1)
    k=x
    df1=df.head(100)
    k.append(y[0])
    global data1
    data1=DataFrame(df1[k])
    click_sub=Button(frame3,text="View",borderwidth="3",width="10",font="5",command=submit)
    click_sub.grid(row=index+2)
    back_2=Button(frame3,text="Back",font="5",width="10",borderwidth="3",command=sub)
    back_2.grid(row=0,column=4)
    option_b=Label(frame3,text="Enter your predictors",font="5")
    option_b.grid(row=4,column=2)
    prediction_button1()
def prediction_button1():
    global ind
    ind=0
    global x_val
    x_val=StringVar()
    global flag
    flag=1
    global fea
    fea=[]
    fun=prediction1()
def prediction1():
    global ind
    lg=Label(frame3,text=x[ind],font="5")
    lg.grid(row=5,column=1)
    
    eg=Entry(frame3,textvariable=x_val)
    eg.grid(row=5,column=2)
    b=Button(frame3,text="click",command=dirs1)
    b.grid(row=5,column=3)
def dirs1():
    global ind
    global fea
    fea.append(float(x_val.get()))
    print(fea)
    if(ind<len(x)-2):
        print(ind)
        flag=1
        x_val.set("")
        
        ind+=1
        prediction1()
    else:
        global subm
        subm=Button(frame3,text="view result",command=view_res1)
        subm.grid(row=6,column=2,pady=5)
def view_res1():
    global subm
    r=[fea]
    re=linr.predict(r)
    global result_lab,bac
    print(re)
    result_lab=Label(frame3,text="Result is "+str(re[0]),font="5")
    subm.destroy()
    result_lab.grid(row=6,column=2)
    bac=Button(frame3,text="Clear",command=clear1)
    bac.grid(row=6,column=3)
def clear1():
    global result_lab,bac
    bac.destroy()
    result_lab.destroy()
    prediction_button1()
    
    
def test_but_lin():
    frame2.destroy()
    global frame3
    frame3=LabelFrame(root ,borderwidth=6 ,text="Test and visualisation",font="25", padx=30 , pady=30,width=1000,height=1000)
    frame3.grid(padx=30,pady=30)
    tl=Label(frame3,text=" * Testiing Statistics:: ",font="15")
    tl.grid(row=1,pady=10,column=0)
    from sklearn .metrics import r2_score
    st1=Label(frame3,text=" * No of Testing Samples:: "+"  * "+str(len(xtest)))
    st1.grid(row=2,column=0)
    from sklearn import metrics
    if(alg=="Linear Regression"):
        st5=Label(frame3,text=" * Testing Accuracy::"+"  * "+str(metrics.r2_score(ytest,linr.predict(xtest))))
        st5.grid(row=3,column=0)
        
        error=Label(frame3,text=" * Mean Squarred Error "+"  * "+str(metrics.mean_squared_error(ytest,linr.predict(xtest))))
        error.grid(row=4,column=0,pady=10)

        
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plots=Label(frame3,text=" * Select your plot ",pady=20,font="15")
    plots.grid(row=5,column=0,pady=10)
    global charts,plot_var,df1
    charts=["line plot","Bar","histplot"]
        
        
    plot_var=StringVar()
  #  plot_var().set(charts)
    plt_box=OptionMenu(frame3,plot_var,*charts)
    plt_box.grid(row=6,column=0)
    k=x
    df1=df.head(100)
    k.append(y[0])
    global data1
    data1=DataFrame(df1[k])
    click_sub=Button(frame3,text="View",borderwidth="3",width="10",font="5",command=submit)
    click_sub.grid(row=7,column=0)
    back_2=Button(frame3,text="Back",font="5",width="10",borderwidth="3",command=sub)
    back_2.grid(row=0,column=4)
    show_b=Label(frame3,text="Enter your predictors",font="5")
    show_b.grid(row=4,column=2)
    prediction_button()
def prediction_button():
    global ind
    ind=0
    global x_val
    x_val=StringVar()
    global flag
    flag=1
    global fea
    fea=[]
    fun=prediction()
def prediction():
    global ind
    lg=Label(frame3,text=x[ind],font="5")
    lg.grid(row=5,column=1)
    
    eg=Entry(frame3,textvariable=x_val)
    eg.grid(row=5,column=2)
    b=Button(frame3,text="click",command=dirs)
    b.grid(row=5,column=3)
    
def dirs():
    global ind
    global fea
    fea.append(float(x_val.get()))
    print(fea)
    if(ind<len(x)-2):
        print(ind)
        flag=1
        x_val.set("")
        
        ind+=1
        prediction()
    else:
        global subm
        subm=Button(frame3,text="view result",command=view_res2)
        subm.grid(row=6,column=2,pady=5)
def view_res2():
    global subm
    r=[fea]
    re=linr.predict(r)
    global result_lab,bac
    result_lab=Label(frame3,text="Result is "+str(re[0][0]),font="5")
    subm.destroy()
    result_lab.grid(row=6,column=2)
    bac=Button(frame3,text="Clear",command=clear)
    bac.grid(row=6,column=3)
def clear():
    global result_lab,bac
    bac.destroy()
    result_lab.destroy()
    prediction_button()
    
    
        
        
    
    
        

    

    
    
    
    

    
    
    
    
def stat1():
    global y1
    tl=Label(frame2,text=" * Training statistics:: ",font="5")
    tl.grid(row=1,pady=10,column=1)
    from sklearn import metrics
    
    st1=Label(frame2,text=" * No of training samples:: "+"  * "+str(len(xtrain)))
    st1.grid(row=6,column=0)
    st3=Label(frame2,text=" * total no of smaples::"+ "  * "+str(len(xtrain)+len(xtest)))
    st3.grid(row=5,column=1)
    st5=Label(frame2,text=" * training accuracy::"+"  * "+str(metrics.accuracy_score(ytrain,linr.predict(xtrain))))
    st5.grid(row=6,column=1)
    st7=Label(frame2,text=" * Confusion Matrix",font="5")
    st7.grid(row=7,column=0)
    import seaborn as sn
    import matplotlib.pyplot as plt
    global cnf_matrix
    cnf_matrix=metrics.confusion_matrix(ytrain,linr.predict(xtrain))
   
# plt.figure(figsize=(10,7))
    print(cnf_matrix)
    c=[]
    cnf_button=Button(frame2,text="Click here to view the confusion matrix",command=view_conf,borderwidth="3")
    cnf_button.grid(row=8,column=0)
    
    st8=Label(frame2,text=" * mean of independent and dependent variables",font="5")
    st8.grid(row=8,column=1,pady=5)
    y2=[]
    for i in range(len(x)):
        y2.append("*"+x[i]+"  *"+str(df[x[i]].mean()))
    for i in range(len(y2)):
        mean=Label(frame2,text=y2[i])
        mean.grid(row=i+10,column=1)
    index=i+11
    ypr=linr.predict(xtrain)
    print(ytrain,ypr)
    f_sco=metrics.precision_recall_fscore_support(ytrain,linr.predict(xtrain),average="weighted")
    pres=Label(frame2,text="* precison ::  "+" * "+str(f_sco[0]))
    pres.grid(row=2,column=1)
    recall=Label(frame2,text=" * recall  ::"+" * "+str(f_sco[1]))
    recall.grid(row=3,column=1)
    fscore=Label(frame2,text="* Fscore ::"+" * "+str(f_sco[2]))
    fscore.grid(row=4,column=1)
    print(f_sco)
    gradients=linr.coef_
    global unique
    global uv
    uniq=df[y].nunique()
    uv=list(uniq)
    print(gradients)
    grade=[]
    for i in range(len(gradients)):
        grade.append(list(gradients[i]))
    grade=np.array(grade)
    print(grade)
    grad=Label(frame2,text=" * gradients/coef :: "+" * "+str(grade))
    grad.grid(row=index+1,column=0)
    inter=Label(frame2,text=" * Intercept :: "+" * "+str(linr.intercept_))
    inter.grid(row=index+1,column=1)
        
    test_b=Button(frame2,text="Test and visualisation",width="20",font="5",padx=80,borderwidth="5",command=test_but)
    test_b.grid(row=index+2,pady=20,padx=100)
    test_l=Label(frame2,text="Click here to validate/test the data")
    test_l.grid(row=index+3)
def view_conf():
    import seaborn as sn
    import matplotlib.pyplot as plt
    global cnf_matrix
    df_cm = pd.DataFrame(cnf_matrix, range(len(cnf_matrix)), range(len(cnf_matrix)))
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    figure = plt.Figure(figsize=(6,5), dpi=100)
    ax = figure.add_subplot(111)
    top=Toplevel()
    chart_type = FigureCanvasTkAgg(figure,top)
    chart_type.get_tk_widget().grid(row=8)
# plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},ax=ax)    
    
def stat():
    global y1
    tl=Label(frame2,text=" * Training statistics:: ",font="5")
    tl.grid(row=5,pady=10,column=0)
    from sklearn .metrics import r2_score
    st1=Label(frame2,text=" * No of training samples:: "+"  * "+str(len(xtrain)))
    st1.grid(row=6,column=0)
    st3=Label(frame2,text=" * total no of smaples::"+ "  * "+str(len(xtrain)+len(xtest)))
    st3.grid(row=6,column=1)
    from sklearn import metrics
    st5=Label(frame2,text=" * training accuracy::"+"  * "+str(metrics.r2_score(ytrain,linr.predict(xtrain))))
    st5.grid(row=7,column=0)
    error=Label(frame2,text=" * Mean Squarred Error "+str(metrics.mean_squared_error(ytrain,linr.predict(xtrain))))
    error.grid(row=7,column=1,pady=10)
    st7=Label(frame2,text=" * mean of independent and dependent variables",font="5")
    st7.grid(row=8,column=0,pady=5)
    y1=[]
    for i in range(len(x)):
        y1.append("*"+x[i]+"  *"+str(df[x[i]].mean()))
    y1.append("*"+y[0]+" *"+str(df[y[0]].mean()))
    for i in range(len(y1)):
        mean=Label(frame2,text=y1[i])
        mean.grid(row=i+9,column=0)
    index=i+9
    st7=Label(frame2,text=" * variance of independent and dependent variables",font="5")
    st7.grid(row=8,column=1)
    y1=[]
    for i in range(len(x)):
        y1.append("*"+x[i]+"  *"+str(df[x[i]].var()))
    y1.append("*"+y[0]+" *"+str(df[y[0]].var()))
    for i in range(len(y1)):
        mean=Label(frame2,text=y1[i])
        mean.grid(row=i+9,column=1)
    index=i+9
    gradients=list(linr.coef_)
    global unique
    global uv
    uniq=df[y].nunique()
    uv=list(uniq)
    print(gradients)
    grad=Label(frame2,text=" * gradients/coef :: "+" * "+str((linr.coef_)))
    grad.grid(row=index+1,column=0)
    inter=Label(frame2,text=" * Intercept :: "+" * "+str(linr.intercept_))
    inter.grid(row=index+1,column=1)
    test_b=Button(frame2,text="Test and visualisation",width="20",font="5",padx=80,borderwidth="5",command=test_but)
    test_b.grid(row=index+2,pady=20,padx=100)
    test_l=Label(frame2,text="Click here to validate/test the data")
    test_l.grid(row=index+3)

def submit():
    print(plot_var.get())
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    chart_type=plot_var.get()
    if(chart_type=="Bar"):
        figure = plt.Figure(figsize=(13,6), dpi=100)
        ax = figure.add_subplot(111)
        top=Toplevel()
        frame4=LabelFrame(top, text=" bar " , padx=5 , pady=5 ,font=("helvetica" ,20) )
        frame4.grid(padx=15 , pady=15)
        chart_type = FigureCanvasTkAgg(figure,frame4)
        chart_type.get_tk_widget().pack()
       # df = df[['First Column','Second Column']].groupby('First Column').sum()
        data1.plot(kind="bar", legend=True, ax=ax)
        
        
    elif(chart_type=="line plot"):

        figure = plt.Figure(figsize=(6,5), dpi=100)
        ax = figure.add_subplot(111)
        top=Toplevel()
        frame4=LabelFrame(top, text=" line plot " , padx=5 , pady=5 ,font=("helvetica" ,20) )
        frame4.grid(padx=15 , pady=15)
        chart_type = FigureCanvasTkAgg(figure,frame4)
        chart_type.get_tk_widget().pack()
       # df = df[['First Column','Second Column']].groupby('First Column').sum()
        data1.plot(kind="line", legend=True, ax=ax)
    elif(chart_type=="precision_recall"):
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import plot_precision_recall_curve

        figure = plt.Figure(figsize=(6,5), dpi=100)
        ax = figure.add_subplot(111)
        top=Toplevel()
        frame4=LabelFrame(top, text="precision -recall " , padx=5 , pady=5 ,font=("helvetica" ,20) )
        frame4.grid(padx=15 , pady=15)
        chart_type = FigureCanvasTkAgg(figure,frame4)
        chart_type.get_tk_widget().pack()
        plot_precision_recall_curve(linr,xtest,ytest,ax=ax)
    else:
        figure = plt.Figure(figsize=(6,5), dpi=100)
        ax = figure.add_subplot(111)
        top=Toplevel()
        frame4=LabelFrame(top, text=" hist plot " , padx=5 , pady=5 ,font=("helvetica" ,20) )
        frame4.grid(padx=15 , pady=15)
        chart_type = FigureCanvasTkAgg(figure,frame4)
        chart_type.get_tk_widget().pack()
       # df = df[['First Column','Second Column']].groupby('First Column').sum()
        data1.plot(kind="hist", legend=True, ax=ax)

        
   
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        
def main():
    dest()
    global k
    k=1
    global bg_image
    bg_image=ImageTk.PhotoImage(Image.open("bg1.jpg"))
    bg_label= Label(root,image=bg_image)
    bg_label.image=bg_image
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    global frame,frame01,frame02
    root.title("ML Prediction and Data Visualization")
    main_frame=LabelFrame(root ,borderwidth=6 ,text="Greenviz",font="25", padx=10 , pady=5,width=1000,height=1000)
    main_frame.grid(padx=200,pady=20)
    frame=LabelFrame(main_frame ,text="Upload",borderwidth="3",font="25", padx=94 , pady=5,width=10000,height=1000)
    frame.grid(padx=10)

    upload_label=Label(frame , text=" * Upload your file *(.csv , .xlsx)",height="1",font="5")
    upload_label.grid(row=1)

    browse_button=Button(frame , text="Browse...here" , command=panfiles)
    browse_button.grid(row=2,pady=2)


    global inde_enter
    select_label=Label(frame , text="*  Select your algorithm")
    select_label.grid(row=4 , column=0)
    clicked.set("*   Select algorithm     *")
    enter_alg=OptionMenu(frame , clicked , "Linear Regression" , "Logistic Regression","Desicion Tree classifier")
    enter_alg.grid(row=4, column=1)
    frame01=LabelFrame(main_frame ,borderwidth="3",text="Feature Selection",font="25", padx=133, pady=5,width=1000,height=1000)
    frame01.grid(padx=10)

    space_label1=Label(frame01 , text="                                                                                   ")
    space_label1.grid(row=6)
    
    feat_label=Label(frame01, text="*Enter your features Below::--")
    feat_label.grid(row=7)

    inde_label=Label(frame01, text="*Enter your Independent Variable(x)   ::")
    inde_label.grid(row=8 , column=0)
    global inde_var,inde_enter
    inde_var=StringVar()
    inde_enter=Listbox(frame01,listvariable=inde_var,selectmode=MULTIPLE,height=3)
    inde_enter.grid(row=8 , column=1)
    global de_var
    de_var=StringVar()
    de_label=Label(frame01, text="*Enter your Dependent Variable (y) ::")
    de_label.grid(row=10 , column=0)
    de_enter=OptionMenu(frame01 ,de_var,"")
    de_enter.grid(row=10, column=1)
    space_label=Label(frame , text="                                                                            ")
    space_label.grid(row=10)
    frame02=LabelFrame(main_frame,text="Train and Test",borderwidth="3",font="25", padx=108,pady="8" ,width=1000,height=1000)
    frame02.grid(padx=10)

    percent_label=Label(frame02 , text="* Enter your train and test percentage Below::")
    percent_label.grid(row=12 )
    
    train_label=Label(frame02, text="Enter your train percentage ::")
    train_label.grid(row=13 ,column=0)
    global train_p
    global test_p
    test_p=StringVar()
    train_p=StringVar()
    train_entry=Entry(frame02, width=30 , borderwidth=3,textvariable=train_p)
    train_entry.grid(row=13 , column=1)
    
    test_label=Label(frame02, text="Enter your test percentage ::")
    test_label.grid(row=15 ,column=0)
    test_entry=Entry(frame02 , width=30 , borderwidth=3,textvariable=test_p)
    test_entry.grid(row=15 , column=1)
    
    button=Button(main_frame , text="Submit",command=sub ,width="10",font="5")
    button.grid(row= 16 , column=0)
    back_l=Button(main_frame,text="Back to login",command=login,width="15",font="10",borderwidth="3")
    back_l.grid(row=17,column=0)
    
def reset():
    messagebox.showinfo("Forgot Password" , "A reset Password link will be sent to your mail")
    
def succregister():
    messagebox.showinfo("Successful!" , "Successfully Registered ,Go to login page and continue")
    login()
def passcom():
    pass
def register():
    dest()
    global bg_image
    bg_image=ImageTk.PhotoImage(Image.open("bg1.jpg"))
    bg_label= Label(root,image=bg_image)
    bg_label.image=bg_image
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    global my_label , space_label , name_label , name_entry , space_label1 , reg_label , reg_entry , space_label2 , mail_label , mail_entry , space_label3 , branch_label , branch_entry , space_label4 , pass_label , pass_entry , space_label5 , forgot_button 
    frame02=LabelFrame(root, text="REGISTER"  , padx=100 , pady=100 , font=("helvetica" , 26))
    frame02.grid(padx=300 , pady=40)

    space_label=Label(frame02, text="                                                     ")
    space_label.grid(row=1)

    name_label=Label(frame02, text="*NAME       ::" , font=("helvetica" , 20))
    name_label.grid(row=2 ,column=0)


    name_entry=Entry(frame02, width=40 , borderwidth=3)
    name_entry.insert(0 ,"Enter your name..." )
    name_entry.grid(row=2 ,column=2)

    space_label1=Label(frame02, text="                                                     ")
    space_label1.grid(row=3)


    reg_label=Label(frame02 , text="*REGISTER NUMBER      ::" , font=("helvetica" , 20))
    reg_label.grid(row=4 , column=0)

    reg_entry=Entry(frame02 , width=40 ,borderwidth=3)
    reg_entry.insert(0 , "Enter your Register number..")
    reg_entry.grid(row=4 , column=2)

    space_label2=Label(frame02, text="                                                     ")
    space_label2.grid(row=5)

    mail_label=Label(frame02 , text="*MAIL.ID      ::" , font=("helvetica" , 20))
    mail_label.grid(row=6 , column=0)


    mail_entry=Entry(frame02 , width=40 ,borderwidth=3)
    mail_entry.insert(0 , "ex: someone@gmail.com")
    mail_entry.grid(row=6 , column=2)

    space_label3=Label(frame02, text="                                                     ")
    space_label3.grid(row=7)


    branch_label=Label(frame02 , text="*BRANCH      ::" , font=("helvetica" , 20))
    branch_label.grid(row=8 , column=0)

    Branch_entry=Entry(frame02 , width=40 ,borderwidth=3)
    Branch_entry.insert(0 , "Enter your Branch(ex: CSE,ECE,...)")
    Branch_entry.grid(row=8 , column=2)

    space_label4=Label(frame02, text="                                                     ")
    space_label4.grid(row=9)

    pass_label=Label(frame02 , text="*PASSWORD    ::" , font=("helvetica" , 20))
    pass_label.grid(row=10 , column=0)

    pass_entry=Entry(frame02 , width=40 ,borderwidth=3)
    pass_entry.insert(0 , "enter your password...")
    pass_entry.grid(row=10 , column=2)

    space_label5=Label(frame02, text="                                                     ")
    space_label5.grid(row=11)
    
    space_label6=Label(frame02, text="                                                     ")
    space_label6.grid(row=12)
    
    
    forgot_button=Button(frame02 , text="Register" ,borderwidth="3", command=succregister,width="10",font="10")
    forgot_button.grid(row=13)
    about_button=Button(frame02,text="Back to Home Page",command=about,width="15",font="5",borderwidth="3")
    about_button.grid(row=14,column=4)

def lin1():
    global k
    if(k==1):
        clicked.set("Linear Regression")
        main()
        
    else:
        messagebox.showerror("Error","please login")
        login()
        
def log1():
    global k
    if(k==1):
        clicked.set("Logistic Regression")
        main()
    else:
        messagebox.showerror("Error","please login")
        login()
    
def login():
    dest()
    
    global bg_image
    bg_image=ImageTk.PhotoImage(Image.open("bg1.jpg"))
    bg_label= Label(root,image=bg_image)
    bg_label.image=bg_image
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    global my_label , space_label , name_label , name_entry , for_button , space_label1 , reg_label , reg_entry , space_label2 , mail_label , mail_entry , space_label3 , branch_label , branch_entry , space_label4 , pass_label , pass_entry , space_label5 , forgot_button 
    frame03=LabelFrame(root , text="LOGIN"  , padx=50 , pady=60 , font=("helvetica" , 24))
    frame03.grid(padx=100 , pady=10)

    reg_label=Label(frame03 , text="*REGISTER NUMBER      ::" , font=("helvetica" , 20))
    reg_label.grid(row=4 , column=0)

    reg_entry=Entry(frame03 , width=40 ,borderwidth=3,font="15")
    reg_entry.insert(0 , "Enter your Register number..")
    reg_entry.grid(row=4 , column=2)


    space_label2=Label(frame03, text="                                                     ")
    space_label2.grid(row=5)
    
    
    space_label3=Label(frame03, text="                                                     ")
    space_label3.grid(row=6)
    
    space_label4=Label(frame03, text="                                                     ")
    space_label4.grid(row=6)

    pass_label=Label(frame03 , text="*PASSWORD    ::" ,font=("helvetica" , 20))
    pass_label.grid(row=10 , column=0)

    pass_entry=Entry(frame03 , width=40 ,borderwidth=3,font="15")
    pass_entry.insert(0 , "enter your password...")
    pass_entry.grid(row=10 , column=2)


    space_label5=Label(frame03, text="                                                     ")
    space_label5.grid(row=11)
    
    space_label6=Label(frame03, text="                                                     ")
    space_label6.grid(row=12)
    
    space_label7=Label(frame03, text="                                                     ")
    space_label7.grid(row=13)

    forgot_button=Button(frame03 , text="Forgot password" , command=reset,width="15",font="10" )
    forgot_button.grid(row=14 , column=1)

    for_button=Button(frame03 , text="Login",width="15",font="10",command=val_log)
    print(k)
    for_button.grid(row=14 , column=2)
    about_button=Button(frame03,text="Back to Home Page",command=about,width="15",font="5",borderwidth="3")
    about_button.grid(row=14,column=4)
def val_log():
    global k
    k=1
    if(k==1):
        print(k)
        main()
    

def dest():
    x1=[]
    for ele in root.winfo_children():
        if ele not in ydes:
            ele.destroy()
        
global x1,alg
x1=[]
root=Tk()
global clicked
clicked=StringVar()
global flag
flag=1
global k
root.geometry("1500x1000")
root.title("ML Prediction and Data Visualization")
global frame0,frame01,frame03,frame02,frame2,top1
k=0
def about():
    global count
    count=0
    dest()

    frame0=LabelFrame(root, text=" About  " , padx=40 , pady=40 ,font=("helvetica" ,20),borderwidth="5" )
    frame0.grid(padx=300 , pady=40)

    des_label1=Label(frame0 , text="Greenviz is an open source Machine Learning and Visualization engine"  ,font=("helvetica",13))
    des_label1.grid(row=1 , column=0)


    des_label2=Label(frame0 , text="Developed by the open source community at Kalasalingam Academy of Research and Education ," ,  font=("helvetica",13))
    des_label2.grid(row=2 , column=0)


    des_label3=Label(frame0 , text=" The Tool Provides an Effective Virtual Lab and environment comprising Lecture videos,lecture" ,  font=("helvetica",13))
    des_label3.grid(row=3 , column=0)

    des_label4=Label(frame0 , text=" materials,assessment and real time machine Learning implementation with appropriate data " ,  font=("helvetica",13))
    des_label4.grid(row=4 , column=0)

    des_label5=Label(frame0 , text="Visualization. The entire tool is authenticated and all the user data are logged " ,  font=("helvetica",13))
    des_label5.grid(row=5 , column=0)

    frame01=LabelFrame(root , text=" Developers " , padx=5 , pady=5 ,font=("helvetica" ,20) ,borderwidth="5")
    frame01.grid(padx=15 , pady=15)



    raja_img=ImageTk.PhotoImage(Image.open("raja12.jpeg"))
    raja_label=Label(frame01, image=raja_img )
    raja_label.image=raja_img
    raja_label.grid(row=0 , column=0)
        
    space_label1=Label(frame01  , text="                                            ")
    space_label1.grid(row=0 , column=1)

    achuth_img=ImageTk.PhotoImage(Image.open("resized achuth pic.jpeg"))
    my_label1=Label(frame01 , image=achuth_img)
    my_label1.image=achuth_img
    my_label1.grid(row=0 , column=2)



    space_label_2=Label(frame01  , text="                                             ")
    space_label_2.grid(row=0 , column=3)

    shiridi_img=ImageTk.PhotoImage(Image.open("shiridi pic rezied.jpeg"))
    my_label_2=Label(frame01 , image=shiridi_img)
    my_label_2.image=shiridi_img
    my_label_2.grid(row=0 , column=4)


    raja_des_label=Label(frame01 , text="Mr.R.Raja Subramanian,")
    raja_des_label.grid(row=1 , column=0)


    raja_des_label_1=Label(frame01 , text="Assistant Professor, CSE, KARE.")
    raja_des_label_1.grid(row=2 , column=0)



    achuth_des_label=Label(frame01 , text="Mr.D.Achuth,")
    achuth_des_label.grid(row=1 , column=2)


    achuth_des_label_1=Label(frame01 , text="UG Scholar, CSE")
    achuth_des_label_1.grid(row=2 , column=2)

    shiridi_des_label=Label(frame01 , text="Mr.P.Shiridi Kumar")
    shiridi_des_label.grid(row=1 , column=4)

    shiridi_des_label_1=Label(frame01 , text="UG Scholor, CSE")
    shiridi_des_label_1.grid(row=2 , column=4)

    new_label=Label(root, text="New User?",font="8",width="20")
    new_label.grid(row=9 , column=0)

    new_button=Button(root , text=" Register" , command=register,font="8",borderwidth="3",width="15" )
    new_button.grid(row=10 , column=0)

    gap_label=Label(root , text="                 ")
    gap_label.grid(row=11 , column=0)


    have_label=Label(root , text="Already Have an Account!",font="8",width="20")
    have_label.grid(row=12 , column=0)

    login_button=Button(root , text="Login" , command=login,font="8",width="15",borderwidth="3")
    login_button.grid(row=13 , column=0)


about()
global count
count=0

my_menu=Menu(root)
global ydes
ydes=[]
ydes.append(my_menu)
ydes.append(k)
print(ydes)
root.config(menu=my_menu)
def reset():
    messagebox.showinfo("Forgot Password" , "A reset Password link will be sent to your mail")
    
def succregister():
    messagebox.showinfo("Successful!" , "Successfully Registered ,Go to login page and continue")


def pdfa():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1PP5oBc3QwwqY46A65YpgBSbxOEgqDCr9/view?usp=sharing")
    
def pdfb():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
    
        webbrowser.open("https://drive.google.com/file/d/15GTRLgRdCdveDb5xUzsDqxvEMCkIlwTE/view?usp=sharing")

def pdfc():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1ZMm3CcvR_qV3XHdVNvSCHY5d8mbweQ8w/view?usp=sharing")
    
def pdfd():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1xUlyyv5n6QUKao3JCY52ivM8dafCIx3U/view?usp=sharing")
    
def pdfe():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1IM8AhNIGnZFLCtxPJiJycKPvZbJIsU5-/view?usp=sharing")
    
def pdff():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1ptvIbkayHX6WCByu6lozYw6oYnISHvHU/view?usp=sharing")
    
def pdfg():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1nXoBS3LIxtnyES_R_LfwtmAe7OVK01wH/view?usp=sharing")
    
def pdfh():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1BU28XI21KKwcujkGW3Dpevu56uECufRg/view?usp=sharing")

def n1():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/19OPnz3nDudHzpU34YARrkl7QY_0bfCug/view?usp=sharing")
    
def n2():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1uPpWh6WqQd4SD_4humIPneVK6z7D2GFw/view?usp=sharing")
def n3():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1BcPFLayvC_7qFC_p4pXCjEga_ce39lUk/view?usp=sharing")
def n4():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/12hyYxhpp6f-Bu0Q5dEzwb58wQYIM6SeT/view?usp=sharing")
def n5():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1zS1WEX9glkb5jkyqXTacFYbpcpTp1GyF/view?usp=sharing")
def n6():
    if(k!=1):
        messagebox.showerror("Error","please login")
        login()
    else:
        webbrowser.open("https://drive.google.com/file/d/1uwrYUdjyVAFbRfsIcCqPWM1i2JR_dAd6/view?usp=sharing")
    


file_menu=Menu(my_menu)
my_menu.add_cascade(label="File" , menu=file_menu)
file_menu.add_command(label="Home",command=about)
file_menu.add_command(label="Login to your Account" , command=login)
file_menu.add_command(label="Register" , command=register)
file_menu.add_separator()
file_menu.add_command(label="Exit" , command=root.quit())



video_menu=Menu(my_menu)

my_menu.add_cascade(label="Video Lectures" , menu=video_menu)
video_menu.add_command(label="Lesson 1" , command=passcom)
video_menu.add_command(label="Lesson 2" , command=passcom)
video_menu.add_command(label="Lesson 3" , command=passcom)
video_menu.add_command(label="Lesson 4" , command=passcom)
video_menu.add_command(label="Lesson 5" , command=passcom)
notebook_menu=Menu(my_menu)
my_menu.add_cascade(label="Jupyter Notebooks" , menu=notebook_menu)

sub_note=Menu(notebook_menu)
notebook_menu.add_cascade(label="Hyper parameters and Model Validation" , menu=sub_note)
sub_note.add_command(label="open" , command=n1)


sub_note1=Menu(notebook_menu)
notebook_menu.add_cascade(label="Support Vector Machines" , menu=sub_note1)
sub_note1.add_command(label="open" , command=n2)


sub_note2=Menu(notebook_menu)
notebook_menu.add_cascade(label="Random Forests" , menu=sub_note2)
sub_note2.add_command(label="open" , command=n3)


sub_note3=Menu(notebook_menu)
notebook_menu.add_cascade(label="Principal-Component-Ananlysis" , menu=sub_note3)
sub_note3.add_command(label="open" , command=n4)

sub_note4=Menu(notebook_menu)
notebook_menu.add_cascade(label="K-Means Clustering" , menu=sub_note4)
sub_note4.add_command(label="open" , command=n5)

sub_note5=Menu(notebook_menu)
notebook_menu.add_cascade(label="KNN and Support Vector Machines" , menu=sub_note5)
sub_note5.add_command(label="open" , command=n6)

    
mat_menu=Menu(my_menu ,font=("helvetica" , 10 , "bold"))
my_menu.add_cascade(label="Lecture Materials" , menu=mat_menu)

submenu=Menu(mat_menu)
mat_menu.add_cascade(label="Introduction to Artificial Intelligence" ,menu=submenu)
submenu.add_command(label="Open" , command=pdfa)

submenu1=Menu(mat_menu)
mat_menu.add_cascade(label="Knowledge Based Systems" , menu=submenu1)
submenu1.add_command(label="Open" ,command=pdfb )

submenu2=Menu(mat_menu)
mat_menu.add_cascade(label="Probablisitic Approach to AI" , menu=submenu2)
submenu2.add_command(label="Open" , command=pdfc)

submenu3=Menu(mat_menu)
mat_menu.add_cascade(label="Evolutionary Intelligence" , menu=submenu3)
submenu3.add_command(label="Open" , command=pdfd)

submenu4=Menu(mat_menu)
mat_menu.add_cascade(label="Neural Networks and NLU" , menu=submenu4)
submenu4.add_command(label="Open" , command=pdfe)

submenu5=Menu(mat_menu) 
mat_menu.add_cascade(label="Introduction to Machine Learning" , menu=submenu5)
submenu5.add_command(label="Open" , command=pdff)

submenu6=Menu(mat_menu)
mat_menu.add_cascade(label="Deterministic Models" , menu=submenu6)
submenu6.add_command(label="Open" , command=pdfg)

submenu7=Menu(mat_menu)
mat_menu.add_cascade(label="Introduction to Python" , menu=submenu7)
submenu7.add_command(label="Open" , command=pdfh)



quiz_menu=Menu(my_menu)
my_menu.add_cascade(label="QUIZ" , menu=quiz_menu)
quiz_menu.add_command(label="QUIZ 1" , command=passcom)
quiz_menu.add_command(label="QUIZ 2" , command=passcom)
quiz_menu.add_command(label="QUIZ 3" , command=passcom)
quiz_menu.add_command(label="QUIZ 4" , command=passcom)
quiz_menu.add_command(label="QUIZ 5" , command=passcom)

lab_menu=Menu(my_menu)
my_menu.add_cascade(label="V lab" , menu=lab_menu)
lab_menu.add_command(label="Linear Regression" , command=lin1)
lab_menu.add_command(label="Logistic Regression" , command=log1)
lab_menu.add_command(label="Random Forest" , command=passcom)
lab_menu.add_command(label="K means clustering" , command=passcom)




if __name__.endswith('__main__'):

    root.mainloop()
    

