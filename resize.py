
# coding: utf-8

# In[ ]:


from PIL import Image
import os


# In[ ]:


pimgs=os.listdir('data/positive')
nimgs=os.listdir('data/negative')


# In[ ]:


print(str(len(pimgs))+" postive images are found ")
print(str(len(nimgs))+" negative images are found ")


# In[ ]:


width=224
height=224


# In[ ]:


for i in range(len(pimgs)):
    img = 'data/positive/'+pimgs[i]
    im1= Image.open(img)
    im2=im1.resize((width,height), Image.NEAREST)
    im2.save('data2/positive/'+pimgs[i]+".jpg")


# In[ ]:


for i in range(len(nimgs)):
    img = 'data/negative/'+nimgs[i]
    im1= Image.open(img)
    im2=im1.resize((width,height), Image.NEAREST)
    im2.save('data2/negative/'+pimgs[i]+".jpg")

