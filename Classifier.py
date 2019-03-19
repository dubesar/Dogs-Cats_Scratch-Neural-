c=[]
from tqdm import tqdm
import cv2
for i in tqdm(os.listdir('../input/train/train/')):
    img = cv2.imread(os.path.join('../input/train/train/',i),cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(96,96))
    c.append(img)
c = np.array(c)
    
label = []
for file in tqdm(os.listdir('../input/train/train/')):
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
label = np.array(label)

c = c.reshape(25000,96*96)
from keras import *
import keras
y_train = keras.utils.to_categorical(label, 2)
c = c.astype('float32')
c/=255
class NN:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.inp_dim = x.shape[1]
        self.out_dim = 2
        self.lr = 0.01
        
        self.w1 = np.random.randn(self.inp_dim,self.out_dim)
    def forward(self):
        self.z1 = np.dot(self.x,self.w1)
        print(self.z1.shape)
        self.a1 = 1/(1+(np.exp(-1*self.z1)))
        print(self.a1.shape)
    def back_prop(self):
        self.v = (self.a1-self.y)/self.x.shape[0]
        self.dw = np.dot(self.x.T,self.v)
        
        self.w1 -= self.lr*self.dw
    def predict_data(self,data):
        self.x = data
        self.forward()
        return self.a1
    
model = NN(c[:20000],y_train[:20000])
for i in range(5):
    model.forward()
    model.back_prop()
    print(i)
print(model.predict_data(c[20006]))
