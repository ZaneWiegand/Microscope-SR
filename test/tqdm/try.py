# %%
from tqdm import trange
from tqdm import tqdm
from time import sleep
# %%
text = ""
for char in tqdm(["a", "b", "c", "d"]):
    sleep(0.25)
    text = text + char
print(text)
# %%
for i in tqdm(range(100)):
    sleep(0.01)
# %%
# 除了tqdm，还有trange,使用方式完全相同
for i in trange(100):
    sleep(0.01)
# %%
# 只要传入list都可以：
pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    sleep(0.25)
    pbar.set_description("Processing %s" % char)
# %%
text = 'abcdefghijklmnopqrstuvwxyz'
TEXT = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
with tqdm(total=len(TEXT)) as pbar:
    for i in range(len(text)):
        sleep(0.1)
        msg = TEXT[i]+text[i]
        sleep(0.1)
        pbar.set_postfix(msg='{}'.format(msg))
        pbar.update(len(msg)//2)
# %%
