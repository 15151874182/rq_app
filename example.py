import tkinter as tk
from tkinter import filedialog, ttk
import csv

def open_csv():
    # 打开文件对话框，选择 CSV 文件
    file_path = filedialog.askopenfilename(
        title="选择 CSV 文件",
        filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
    )
    
    if file_path:
        # 显示文件路径
        path_label.config(text=f"已选择文件：{file_path}")
        
        # 清空表格
        for row in treeview.get_children():
            treeview.delete(row)
        
        # 读取 CSV 文件
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader)  # 读取表头
                
                # 设置表格列
                treeview["columns"] = headers
                treeview["show"] = "headings"
                
                # 设置表头
                for col in headers:
                    treeview.heading(col, text=col)
                    treeview.column(col, width=100, anchor='w')
                
                # 插入数据
                for row in csv_reader:
                    treeview.insert("", "end", values=row)
                    
        except Exception as e:
            path_label.config(text=f"错误：{str(e)}")

# 创建主窗口
root = tk.Tk()
root.title("微盘股择时评估app")
root.geometry("800x600")

# 创建工具栏框架
toolbar = tk.Frame(root)
toolbar.pack(pady=10, fill='x')

# 添加目录图标按钮 (使用系统图标)
open_icon = tk.PhotoImage(file="")  # 这里可以替换为你的图标文件路径
open_button = ttk.Button(
    toolbar,
    text="打开 CSV 文件",
    command=open_csv,
    compound='left'
)
open_button.pack(side='left', padx=5)

# 显示文件路径的标签
path_label = tk.Label(root, text="未选择文件", anchor='w')
path_label.pack(fill='x', padx=10, pady=5)

# 创建表格显示区域
tree_frame = tk.Frame(root)
tree_frame.pack(fill='both', expand=True, padx=10, pady=5)

# 创建带滚动条的表格
scrollbar = ttk.Scrollbar(tree_frame)
scrollbar.pack(side='right', fill='y')

treeview = ttk.Treeview(
    tree_frame,
    yscrollcommand=scrollbar.set,
    selectmode='browse'
)
treeview.pack(fill='both', expand=True)
scrollbar.config(command=treeview.yview)

# 运行主循环
root.mainloop()