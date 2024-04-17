import tkinter as tk
import cv2
import mediapipe as mp
from tkinter import ttk
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import threading

mutex = threading.Lock()

class window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('BlazePose Research Ver 1.0')
        self.geometry('1080x720')
        #menubar
        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='文件', menu=self.filemenu)
        self.filemenu.add_command(label='新建', command=self.do_job,state = tk.DISABLED)
        self.filemenu.add_command(label='打开', command=self.do_job,state = tk.DISABLED)
        self.filemenu.add_command(label='保存', command=self.do_job,state = tk.DISABLED)
        self.filemenu.add_separator()    # 添加一条分隔线
        self.filemenu.add_command(label='退出', command=window.quit,state = tk.DISABLED) # 用tkinter里面自带的quit()函数
        
        self.editmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='编辑', menu=self.editmenu)
        self.editmenu.add_command(label='剪切', command=self.do_job,state = tk.DISABLED)
        self.editmenu.add_command(label='复制', command=self.do_job,state = tk.DISABLED)
        self.editmenu.add_command(label='黏贴', command=self.do_job,state = tk.DISABLED)
        
        self.possmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='体位', menu=self.possmenu)
        self.possmenu.add_command(label='手(ESC退出)', command=self.do_hand)
        self.possmenu.add_command(label='关节(ESC退出)', command=self.do_poss)
    
        self.analysismenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='数据分析', menu=self.analysismenu)
        self.analysismenu.add_command(label='傅里叶变换(仅第一个点)', command=self.draw_fft,state = tk.DISABLED)
        self.analysismenu.add_command(label='位置(仅第一个点)', command=self.draw_poss,state = tk.DISABLED)
        self.analysismenu.add_command(label='直方图(仅第一个点)', command=self.draw_his,state = tk.DISABLED)
        self.analysismenu.add_command(label='箱线图(仅第一个点)', command=self.draw_box,state = tk.DISABLED)
        
        self.helpmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='帮助', menu=self.helpmenu)
        self.helpmenu.add_command(label='关于', command=self.helpWindow)
   
        self.submenu = tk.Menu(self.filemenu) # 和上面定义菜单一样，不过此处实在File上创建一个空的菜单
        self.filemenu.add_cascade(label='输入', menu=self.submenu, underline=0) # 给放入的菜单submenu命名为Import
        self.submenu.add_command(label='python文档', command=self.do_job,state = tk.DISABLED)   # 这里和上面创建原理也一样，在Import菜单项中加入一个小菜单命令Submenu_1
        self.config(menu=self.menubar)
        #存储显示用的数据
        self.dataX = []
        self.dataY = []
        self.dataZ = []
        
        #显示文字
        label = tk.Label(self, text="注意:本软件仅用于测试google mediapipe")
        label.pack()
        label = tk.Label(self, text="使用方法：")
        label.pack()
        label = tk.Label(self, text="1.安装USB摄像头(1套)")
        label.pack()
        label = tk.Label(self, text="2.安装python-3.11.3-amd64.exe")
        label.pack()
        label = tk.Label(self, text="3.安装cv2库")
        label.pack()
        label = tk.Label(self, text="4.安装mediapipe库")
        label.pack()
        label = tk.Label(self, text="注意：在体位菜单下选择需要分析的体位获得数据,窗口关闭按ESC键才能退出,窗口关闭后再进入数据分析菜单进行数据解析，因为软件尚未完善，先停掉数据输入才能进入分析状态")
        label.pack()
        label = tk.Label(self, text="2023-4-14 陈华")
        label.pack()
    def do_hand(self):
        self.DisableMenu()
        # 导入solution
        self.dataX.clear()
        self.dataY.clear()
        self.dataZ.clear()
        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        # 打开摄像头
        mutex.acquire()
        cap = cv2.VideoCapture(0)
        winname = "hand"
        idx = 0
        cv2.namedWindow(winname,cv2.WINDOW_AUTOSIZE) 
        #禁用
        while True:
            flag, frame = cap.read()
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            results = hands.process(frameRGB)
            if results.multi_hand_landmarks:
                idx+=1
                # print(results.multi_hand_landmarks)
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if((id == 0)and(idx<1000)) :
                            self.dataX.append(lm.x )
                            self.dataY.append(lm.y)
                            self.dataZ.append(lm.z)
                        print(f'Landmark {id} - (x={lm.x}, y={lm.y}, z={lm.z})')
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            cv2.imshow(winname, frame)
            k = cv2.waitKey(1)
            #按esc键或q键退出死循环
            if k == 27:
                self.enableMenu()
                break
            elif k == ord('q'):
                self.enableMenu()
                break
        cap.release()
        cv2.destroyAllWindows()
        mutex.release()
        
    def do_poss(self):
        self.dataX.clear()
        self.dataY.clear()
        self.dataZ.clear()
        self.DisableMenu()
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mutex.acquire()
        cap = cv2.VideoCapture(0)
        mp_drawing = mp.solutions.drawing_utils
        idx = 0
        while True :
            ret, frame = cap.read()
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            results = pose.process(frameRGB)
            if results.pose_landmarks:
                # print(results.pose_landmarks)
                idx+=1
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if((id == 0)and(idx<1000)) :
                        self.dataX.append(lm.x )
                        self.dataY.append(lm.y)
                        self.dataZ.append(lm.z)
                        print(f'Landmark {idx} - (x={lm.x}, y={lm.y}, z={lm.z})')
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
            cv2.imshow('poss', frame)
            k = cv2.waitKey(1)
            if k == 27:
                self.enableMenu()
                break
            elif k == ord('q'):
                self.enableMenu()
                break
        cap.release()
        cv2.destroyAllWindows()
        mutex.release()
    def draw_fft(self):
        x = self.dataX
        y = self.dataY
        z = self.dataZ 
        if len(x)>0:
            signal = x
            T = 1/ len(x)  # 采样间隔
            t = np.arange(0, 1, T)  # 时间向量
            
            # 傅里叶变换
            fft_signal = np.fft.fft(signal)
            fft_freqs = np.fft.fftfreq(len(fft_signal), T)
            
            # 计算幅度谱
            magnitude_spectrum = np.abs(fft_signal)
            
            # 绘制时域图
            plt.subplot(2, 1, 1)
            plt.plot(t, signal)
            plt.title("Time Domain Signal")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            
            # 绘制频域图
            plt.subplot(2, 1, 2)
            plt.stem(fft_freqs[:len(fft_freqs)//2], magnitude_spectrum[:len(fft_freqs)//2], 'b', markerfmt=" ", basefmt="-b")
            plt.title("Frequency Domain Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.show()   
    
        
    def draw_box(self):
        x = self.dataX
        y = self.dataY
        z = self.dataZ 
        if len(x)>0:
            # 箱线
            data = [x, y, z]
            plt.boxplot(data)

            # 设置图表属性
            plt.title('box')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend(['x','y','z'])

            # 显示图表
            plt.show()   
    def draw_his(self):
        x = self.dataX
        y = self.dataY
        z = self.dataZ 
        if len(x)>0:
            # 绘制直方图
            plt.hist(x, bins=30, alpha=0.5, label='x')
            plt.hist(y, bins=30, alpha=0.5, label='y')
            plt.hist(z, bins=30, alpha=0.5, label='z')

            # 设置图表属性
            plt.title('hist')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()

            # 显示图表
            plt.show()
    def draw_poss(self):
        x = self.dataX
        y = self.dataY
        z = self.dataZ
        if len(x)>0:
            time = np.arange(0,len(x))
            mean ="average value:(" + str(f'{np.mean([x]): .2f}')+','+ str(f'{np.mean([y]): .2f}')+','+ str(f'{np.mean([z]): .2f}')+")\n"
            var = "variance value:(" + str(f'{np.var([x]): .2f}')+','+ str(f'{np.var([y]): .2f}')+','+ str(f'{np.var([z]): .2f}')+")\n"
            std = "standard deviation:(" + str(f'{np.std([x]): .2f}')+','+ str(f'{np.std([y]): .2f}')+','+ str(f'{np.std([z]): .2f}')+")"
            print("平均值:",mean)
            print("方差:", var)
            print("标准差:", std)
            fig,ax = plt.subplots() # 创建一个图fig， 默认包含一个axes
            ax.plot(time,x) # 绘制x-y的折线图
            ax.plot(time,y) # 绘制x-y的折线图
            ax.plot(time,z) # 绘制x-y的折线图
            
            # 创建一个简单的图形
            plt.xlabel('time')
            plt.ylabel('poss')
            plt.legend(['x','y','z'])
                    
            # 在图形外部添加文本
            plt.figtext(0.5, 0.95, mean + var + std, ha='center', va='center', size=10)
            
            plt.show()
    def do_job(self,str):
        self.l.config(text=str)
     # 新窗口
    def helpWindow(self, event=None):
        self.newWin = tk.Toplevel()                 # 创建新窗口
        self.newWin.title("关于")                   # 窗口标题
        self.newWin.geometry('600x400+450+240')     # 窗口像素大小(600x400)及显示位置(450+240)
        self.newWin.focus_set()                     # 设置焦点
        # 标签
        self.dataLabel = ttk.Label(self.newWin, text=self.text.get('1.0', tk.END), anchor='nw', relief=tk.GROOVE)
        self.dataLabel.pack(fill=tk.BOTH, expand=True)
        # 关闭
        self.closeButton = ttk.Button(self.newWin, text='关闭', command=self.newWin.destroy)
        self.closeButton.pack(side=tk.BOTTOM)
        
    def DisableMenu(self):
        self.possmenu.entryconfig(0,state = tk.DISABLED)
        self.possmenu.entryconfig(1,state = tk.DISABLED)
        self.analysismenu.entryconfig(0,state = tk.DISABLED)
        self.analysismenu.entryconfig(1,state = tk.DISABLED)
        self.analysismenu.entryconfig(2,state = tk.DISABLED)
        self.analysismenu.entryconfig(3,state = tk.DISABLED)
    
    def enableMenu(self):
        self.possmenu.entryconfig(0,state = tk.NORMAL)
        self.possmenu.entryconfig(1,state = tk.NORMAL)
        self.analysismenu.entryconfig(0,state = tk.NORMAL)
        self.analysismenu.entryconfig(1,state = tk.NORMAL)
        self.analysismenu.entryconfig(2,state = tk.NORMAL)
        self.analysismenu.entryconfig(4,state = tk.NORMAL)

ui = window()
ui.mainloop()    # 循环窗口