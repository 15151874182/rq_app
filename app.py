import tkinter as tk
from tkinter import filedialog, ttk
import csv

# def open_csv():
#     # 打开文件对话框，选择 CSV 文件
#     file_path = filedialog.askopenfilename(
#         title="选择 CSV 文件",
#         filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
#     )
    
#     if file_path:
#         # 显示文件路径
#         path_label.config(text=f"已选择文件：{file_path}")
        
#         # 清空表格
#         for row in treeview.get_children():
#             treeview.delete(row)
        
#         # 读取 CSV 文件
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 csv_reader = csv.reader(file)
#                 headers = next(csv_reader)  # 读取表头
                
#                 # 设置表格列
#                 treeview["columns"] = headers
#                 treeview["show"] = "headings"
                
#                 # 设置表头
#                 for col in headers:
#                     treeview.heading(col, text=col)
#                     treeview.column(col, width=100, anchor='w')
                
#                 # 插入数据
#                 for row in csv_reader:
#                     treeview.insert("", "end", values=row)
                    
#         except Exception as e:
#             path_label.config(text=f"错误：{str(e)}")

# 创建主窗口
root = tk.Tk()
root.title("微盘股择时评估app")
w=1366
h=768
root.geometry(f"{w}x{h}")
root.resizable(width=False, height=False)
canvas = tk.Canvas(root, width=w, height=h)
canvas.pack()

# 第一行
label1 = tk.Label(root, text="crowd", borderwidth=1, relief="raised", font=("Arial", 16, "bold"))
label2 = tk.Label(root, text="macd", borderwidth=1, relief="raised", font=("Arial", 16, "bold"))
label1.place(x=w*0.25-50,y=10,width=100)
label2.place(x=w*0.75-50,y=10,width=100)

# 在画布上画直线
line1 = canvas.create_line(w/2, h*0.6, w/2, 0, fill="black", width=1)
line2 = canvas.create_line(0, h*0.6, w, h*0.6, fill="black", width=1)

# # 第二行
# label3 = tk.Label(root, text="两市成交量")
# label4 = tk.Label(root, text="两市成交量")
# label3.grid(row=1, column=0)
# label4.grid(row=1, column=1)
# label5 = tk.Label(root, text="a dataframe")
# label5.grid(row=1, column=2, rowspan=2)

# # 第三行
# label6 = tk.Label(root, text="拥挤度")
# label7 = tk.Label(root, text="拥挤度\n历史百分位")
# label6.grid(row=2, column=0)
# label7.grid(row=2, column=1)

# # 第四行
# label8 = tk.Label(root, text="结论")
# label9 = tk.Label(root, text="月份")
# label10 = tk.Label(root, text="拥挤度")
# label11 = tk.Label(root, text="macd")
# label8.grid(row=3, column=0)
# label9.grid(row=3, column=1)
# label10.grid(row=4, column=0)
# label11.grid(row=4, column=1)

# 运行主循环
root.mainloop()