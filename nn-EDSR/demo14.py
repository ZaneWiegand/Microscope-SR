# 文件重命名批量操作-2
# %%
import os

path = "./weight_output"
filename_list = os.listdir(path)
# %%
for item in filename_list:
    if item == '.DS_Store':
        filename_list.remove(item)
# %%
item = filename_list[0]
print(item)
# %%
for item in filename_list:
    print(item.split(".")[0]+'.'+item.split(".")[2])
# %%
Alter = item.split("_lr")[0]+'.'+item.split(".")[-1]
print(Alter)
# %%
i = 0
for item in filename_list:
    used_name = path + "/" + item
    Alter = item.split("_lr")[0]+'.'+item.split(".")[-1]
    new = Alter
    # X = item.split('-')[2].split('.')[0]
    # Y = item.split('.')[-1]
    # X = item.split("_")[1].split("第")[-1].split("话")[0].split("卷")[0]
    new_name = path + "/" + new  # 保留原后缀
    os.rename(used_name, new_name)
    print("%s\n重命名为\n%s" % (used_name.split("/")[-1], new_name.split("/")[-1]))
# %%
path = "./weight_output_syn"
filename_list = os.listdir(path)
# %%
for item in filename_list:
    if item == '.DS_Store':
        filename_list.remove(item)
# %%
item = filename_list[0]
print(item)
# %%
for item in filename_list:
    print(item.split(".")[0]+'.'+item.split(".")[2])
# %%
Alter = item.split("_lr")[0]+'.'+item.split(".")[-1]
print(Alter)
# %%
i = 0
for item in filename_list:
    used_name = path + "/" + item
    Alter = item.split("_lr")[0]+'.'+item.split(".")[-1]
    new = Alter
    # X = item.split('-')[2].split('.')[0]
    # Y = item.split('.')[-1]
    # X = item.split("_")[1].split("第")[-1].split("话")[0].split("卷")[0]
    new_name = path + "/" + new  # 保留原后缀
    os.rename(used_name, new_name)
    print("%s\n重命名为\n%s" % (used_name.split("/")[-1], new_name.split("/")[-1]))
# %%
