#!/usr/bin/env python
# coding: utf-8

# # Classificator fine tuning pre-trained CNN 
# Code from [here](https://medium.com/pythons-gurus/classification-of-medical-images-should-you-build-a-model-from-scratch-or-use-transfer-learning-140e94599ae8), [here](https://rumn.medium.com/custom-pytorch-image-classifier-from-scratch-d7b3c50f9fbe) and [here](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)

#In[]:


from PIL import Image
import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import transforms


#In[]:


import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


#In[]:


r_size = 256
input_size_1=1408
input_size_2=81
hidden_size=152
RESIZE=288


# ### Data loader

#In[]:


import os
import pandas as pd
from sklearn.model_selection import train_test_split
# create a csv file with image_path and respective label
image_path_list = []
label_list = []
DATA_F = "classi"
for class_cat in os.listdir(DATA_F):
    print(class_cat)
    for image_object in os.listdir(f"{DATA_F}/{class_cat}"):
        image_path_list.append(f"{DATA_F}/{class_cat}/{image_object}")
        label_list.append(f"{class_cat}")

df = pd.DataFrame()

df["image_path"] = image_path_list
df["label"] = label_list


# now split this main data to train and test
# Define the split ratio
test_ratio = 0.20 # 20% of data will go to test

# split the data
train_df, test_df = train_test_split(df, test_size=test_ratio, 
                                          stratify=df['label'], 
                                          random_state=42)


print(f"Original dataset shape: {df.shape}")
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")


#In[]:


train_df.tail()


#In[]:


for cat in train_df.label.unique().tolist():
    elements = len(train_df.loc[train_df["label"] == cat])
    print(f"{cat} : {elements} items")


#In[]:


class CustomTrainingData(torch.utils.data.Dataset):
    def __init__(self, csv_df, class_list, transform=None):
        self.df = csv_df
        self.transform = transform
        self.class_list = class_list

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.df.iloc[index].image_path).convert('RGB')
        label = self.class_list.index(self.df.iloc[index].label)

        if self.transform:
            image = self.transform(image)

        return image, label


#In[]:


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

t_list = [
        transforms.Resize((RESIZE,RESIZE)),
        transforms.RandomResizedCrop(RESIZE,scale=(0.75,1.00)), # Attempt to Data Augmentation, for few shoots learning
        transforms.RandomRotation(degrees=20), # +- 20 degrees
        transforms.RandomPerspective(distortion_scale=0.1), # can be between 0 and 1
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
]
transform = transforms.Compose(t_list)

transform_1 = transforms.Compose([t_list[0],t_list[5],t_list[6]])

# Create datasets for training & validation, download if necessary
training_set = CustomTrainingData(train_df, train_df.label.unique().tolist(), transform)
validation_set = CustomTrainingData(test_df, test_df.label.unique().tolist(), transform_1)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = tuple(train_df.label.unique().tolist())


#In[]:


sample_img,sample_label = next(iter(training_set))
(transforms.ToPILImage()(sample_img)).show()


#In[]:


classes


#In[]:


NUM_CLASSES = len(classes)


# #### Demo steps data augmentation

#In[]:


transform_demo_img = Image.open(train_df.iloc[0].image_path).convert('RGB')

# Define a list of transformations
transform_pipeline = t_list[:-2]

# Function to apply transformations step-by-step
def demo_visualize_transforms(image, transforms_list):
    images = [image]  # Store intermediate images
    transformed_image = image

    step = 0
    for t in transforms_list:
        transformed_image = t(transformed_image)
        print(f"Done Transform step {step}")
        step+=1
        images.append(transformed_image)

    return images


# Apply transformations step-by-step
demo_transformed_images = demo_visualize_transforms(transform_demo_img, transform_pipeline)

step = 0 
for demo_i in demo_transformed_images:
    print(f"Show step {step}")
    #transforms.ToPILImage()(demo_i).show()
    demo_i.show()
    step +=1


# ## Reference

#In[]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


# Loading pre-trained models. The code below demonstrates the initialization for each model that I use for transfer learning:

# ```python
# # AlexNet:
# from torchvision.models import alexnet, AlexNet_Weights
# weights = AlexNet_Weights.DEFAULT
# model = alexnet(weights=weights).to(device)
# # don't foget to call 'eval' to use the model for inference
# model.eval()
# ```

# ```python
# # VGG16:
# from torchvision.models import vgg16, VGG16_Weights
# weights = VGG16_Weights.DEFAULT
# model = vgg16(weights=weights).to(device)
# # don't foget to call 'eval' to use the model for inference
# model.eval()
# ```

# ```python
# # ResNet<X>, <X> is one of the following: 18, 50, 101
# from torchvision.models import resnet<X>, ResNet<X>_Weights
# weights = ResNet<X>_Weights.DEFAULT
# model = resnet<X>(weights=weights).to(device)
# # don't foget to call 'eval' to use the model for inference
# model.eval()
# ```

#In[]:


# EfficientNet_B<Y>, <Y> is one of the following: 0, 1, 2, 3
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
weights = EfficientNet_B2_Weights.DEFAULT
pretrained = efficientnet_b2(weights=weights).to(device)
# don't foget to call 'eval' to use the model for inference
pretrained.eval()


# The following lines of code demonstrate how to use the pre-trained model to obtain the final output:

#In[]:


img = Image.open("./0001_2_1881_015.jpg")
#img.show()


#In[]:


img1 = transforms.ToTensor()(img)
img1 = transforms.Resize((256, 256))(img1)
s, m = torch.std_mean(img1, dim=(0, 1, 2))
img1 = transforms.Normalize(m, 2*s)(img1)


#In[]:


preprocess = weights.transforms()
x = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    outputs = pretrained(x)
# vector of class-probabilities:
prediction = outputs.squeeze(0).softmax(0)


#In[]:


[weights.meta["categories"][x.item()] for x in prediction.topk(5)[1]]


#In[]:


class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")


# We don’t need the final output; we need the hidden state from the feature extraction block. For AlexNet, VGG16, and EfficientNet_B<any_model>, it is quite straightforward: the model.features() function returns the output of the feature extraction block. The code block below shows how to prepare an input image for my classifier when using AlexNet, VGG16, or EfficientNet_B<any_model> for transfer learning:

#In[]:


with torch.no_grad():
    y = pretrained.features(x)
y.shape


#In[]:


y = y.squeeze(0)
y.shape


#In[]:


# x.shape: x[0] - number of features, x[1], x[2] - picture sizes; 3 dimensions 
y = torch.reshape(y, (y.shape[0], y.shape[1]*y.shape[2])) # 2 dimensions in the result
y.shape


# Note: I use x.squeeze(0) to remove the batch dimension for a single image, as I assume it will be sent to torch-DataLoader, which adds the batch dimension to image batches. Additionally, I reshape x to combine the image size into a single dimension.

# The table below shows how the resolution of the input image is transformed into the resolution of the features tensor and subsequently into the resolution of the input for my classifier, depending on the model:

# | model           | The resolution of the  input image | The resolution of the output from the feature extraction block | The resolution of the input for my classifier |
# |-----------------|------------------------------------|----------------------------------------------------------------|-----------------------------------------------|
# | alexnet         | (224,224)                          | (256,6,6)                                                      | (256,36)                                      |
# | vgg16           | (224,224)                          | (512,7,7)                                                      | (512,49)                                      |
# | resnet18        | (224,224)                          | (512,7,7)                                                      | (512,49)                                      |
# | resnet50        | (224,224)                          | (2048,7,7)                                                     | (2048,49)                                     |
# | resnet101       | (224,224)                          | (2048,7,7)                                                     | (2048,49)                                     |
# | efficientnet_b0 | (224,224)                          | (1280,7,7)                                                     | (1280,49)                                     |
# | efficientnet_b1 | (240,240)                          | (1280,8,8)                                                     | (1280,64)                                     |
# | efficientnet_b2 | (288,288)                          | (1408,9,9)                                                     | (1408,81)                                     |
# | efficientnet_b3 | (300,300)                          | (1536,10,10)                                                   | (1536,100)                                    |

# Let’s look at the classifier that processes the features obtained by the pre-trained model. For all models, I apply the same architecture, adapted to the input sizes and maintaining approximately the same number of trainable parameters (~440,000). The following code block demonstrates the implementation of the classifier:

# #### _hide_

#In[]:


c='''
class DocClassifier(nn.Module):
    def __init__(self):

        super(DocClassifier, self).__init__()
        nc = 24
        nc2 = nc * 2
        nc4 = nc * 4
        sz = int(r_size/32)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, nc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(nc, nc2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nc2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.2),

            nn.Conv2d(nc2, nc4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nc4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.3),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(sz*sz*nc4, nc2),
            nn.ReLU(inplace=True),
            nn.Linear(nc2, 2),
        )

    def forward(self, x):
        out1 = self.cnn1(x)
        out1 = torch.flatten(out1, 1)
        output = self.fc1(out1)
        return output
'''


#In[]:


c='''
class DocClassifierTransfer(nn.Module):
    def __init__(self, num_classes, input_size_1, input_size_2, hidden_size):
        super(DocClassifierTransfer, self).__init__()
        self.num_classes = num_classes

        self.ln1 = nn.Linear(input_size_2, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(input_size_1*hidden_size, self.num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = nn.Flatten()(x)
        x = self.ln2(x)

        return x

net = DocClassifierTransfer(NUM_CLASSES, input_size_1, input_size_2, hidden_size).to(device)
'''


# #### Model

#In[]:


class DocClassifierTransfer(nn.Module):
    def __init__(self, num_classes, input_size_1, input_size_2, hidden_size):
        super(DocClassifierTransfer, self).__init__()
        self.num_classes = num_classes

        self.ln1 = nn.Linear(input_size_2, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(input_size_1*hidden_size, self.num_classes)
        self.dropout = nn.Dropout(p=0.2)

        self.flat1 = nn.Flatten()

        weights_pre = EfficientNet_B2_Weights.DEFAULT
        self.pretrained = efficientnet_b2(weights=weights_pre).to(device)
        for p in self.pretrained.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.pretrained.features(x)
            x = torch.reshape(x, (x.shape[0],x.shape[1], x.shape[2]*x.shape[3]))
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.flat1(x)
        x = self.ln2(x)

        return x

model = DocClassifierTransfer(NUM_CLASSES, input_size_1, input_size_2, hidden_size).to(device)


# `<input_size_1>` and `<input_size_2>` are the input resolution values for my classifier. Refer to the right column of Table 2 for more details.
# 
# `<hidden_size>` is a parameter that maintains the number of trainable parameters in my classifier at approximately 440,000.
# 
# The table below shows the `<hidden_size>` values depending on the model:
# 
# | model           | hidden size |
# |-----------------|-------------|
# | alexnet         | 750         |
# | vgg16           | 410         |
# | resnet18        | 410         |
# | resnet50        | 105         |
# | resnet101       | 105         |
# | efficientnet_b0 | 168         |
# | efficientnet_b1 | 168         |
# | efficientnet_b2 | 152         |
# | efficientnet_b3 | 138         |

# A few words about the classifier architecture: I found that the model with approximately 440,000 parameters showed the best performance on the test set. Decreasing or increasing the number of parameters led to lower performance. Note that for transfer learning based on ViT, I used a classifier with a similar architecture and a compatible number of parameters as described above.

# An example summary of the classifier that uses EfficientNet_B1 features as input:

#In[]:


summary (model=model, input_data=next(iter(training_loader))[0].to(device), col_names=['input_size', 'output_size', 'num_params', 'trainable'])


# I use the Adam optimizer with a learning rate of 0.001. After training, I selected the checkpoints that showed the best performance on the test set for all models.

# ## Classic pytorch training loop

# ### Optimizer and Loss function

#In[]:


#loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#In[]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# ### Per batch

#In[]:


def train_one_epoch(epoch_index):#, tb_writer):
    running_loss = 0.
    last_loss = 0.

    DATA_AUGMENTATION = 5 # Artificially augment data of 5 times
    for times in range(0,DATA_AUGMENTATION):
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):

            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            '''
            print(len(data))
            print(data[0].shape)
            print(data[1].shape)
            '''
            every_n = min(len(training_loader.dataset)/10,1000)
            every_n = int(every_n)

            if i % every_n == (every_n - 1 ):
                last_loss = running_loss / (every_n) # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                #tb_x = epoch_index * len(training_loader) + i + 1
                #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        print("  --")

    return last_loss


# ### Per Epoch

#In[]:


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)#, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)

            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    '''
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()
    '''

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_scripted_{}_{}.pt'.format(timestamp, epoch_number)
        #torch.save(model.state_dict(), model_path)
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(model_path) # Save

    epoch_number += 1


#In[]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file; this assumes the first row has the series names.
df = pd.read_csv('data.csv',delimiter=";")

# Create a new figure.
plt.figure(figsize=(10, 6))

# Plot each series. Here we assume that the x-axis can simply be the row index.
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# Place the legend at the top of the plot.
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(df.columns))

# Set the title of the plot.
plt.title("Error per epoch (log 10)")

# Use a logarithmic scale (base 10) for the y-axis.
plt.yscale('log', base=10)

# Optional: Label the axes.
plt.xlabel('Index')
plt.ylabel('Value')

# Adjust layout so everything fits nicely.
plt.tight_layout()

# Display the plot.
plt.show()


# #### Tip : Try change base pre-trained model (did not find any)

# ## Classification using multimodal info

#In[]:


get_ipython().system('pip install kraken')


#In[]:


get_ipython().system('kraken list')


#In[]:


KRAKEN_MODEL="10.5281/zenodo.10592716"
TRANSCRIPTION="krakenout.txt"
KRAKENGET_OUT="krakengetout"
MODELNAME_FILE = "modelname"


#In[]:


get_ipython().system('kraken show {KRAKEN_MODEL}')


#In[]:


get_ipython().system('kraken get {KRAKEN_MODEL} 1> {KRAKENGET_OUT}')


#In[]:


get_ipython().system('cat $KRAKENGET_OUT | sed "s/\\x1B\\[[0-9;]\\{1,\\}[A-Za-z]//g"')


#In[]:


get_ipython().system('cat {KRAKENGET_OUT} | grep "Model name" | sed -e "s/Model name://" >{MODELNAME_FILE}')


#In[]:


get_ipython().system('kraken -d cuda:0 -i 0001_2_1881_015.jpg {TRANSCRIPTION} segment -bl ocr -m $(cat {MODELNAME_FILE})')


#In[]:


get_ipython().system('pip install nltk')


#In[]:


get_ipython().system('rm {KRAKENGET_OUT}')
get_ipython().system('rm {MODELNAME_FILE}')


#In[]:


enumerate("parola")
parola[2]


#In[]:


with open(TRANSCRIPTION,"r") as fd:
    content = fd.read()
content


#In[]:


content0 = ''.join([char if (char.isalpha()) or char == '\n' or char == ' ' or i == (len(content)-1) or content[i+1] == '\n' else '' for (i,char) in enumerate(content)])
content0


#In[]:


content2 = ''.join([char if not ( char == '\n' and content0[i-1].isalpha() ) else ' ' for (i,char) in enumerate(content0)])
content2


#In[]:


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')


#In[]:


from nltk.corpus import wordnet
from nltk.corpus import stopwords


#In[]:


get_ipython().system('python -m spacy download it_core_news_sm')


#In[]:


import spacy


#In[]:


nlp = spacy.load('it_core_news_sm')


#In[]:


def lemma_recursive(p):
    before = p
    update = ""
    while True:
        tokens = nlp(before)
        if len(tokens) > 0:
            update = nlp(before)[0].lemma_
        if update == before:
            break
        before = update
    return update


#In[]:


nlp("lita")[0].lemma_


#In[]:


lemma_recursive("lita")


#In[]:


wordnet.lemmas("lita",lang="ita")


#In[]:


def reverse_str(s):
    return s[::-1]


#In[]:


compound = []
for w in content2.split(' '):
    if any(map(lambda c : not c.isalnum(),w)):
        letters = [c if c.isalpha() else ' ' for c in w]
        #print(f" {list(w)} -> {letters}")
        parts = ''.join(letters).split()
        #print(parts)
        if len(parts)>2:
            continue
        compound += parts
        compound.append(''.join(parts))
compound


#In[]:


def word_exists(p):
    return len(wordnet.lemmas(lemma_recursive(p),lang="ita"))>0


#In[]:


good_words = []
for w in content2.split() + compound:

    if w in stopwords.words("italian"):
        continue
    if word_exists(w) :
        good_words.append(w)
    else :
        for old,new in [('a','à'),('e','è'),('e','é'),('i','ì'),('o','ò'),('u','ù')]:
            if old in w or new in w :
                replace1 = reverse_str(reverse_str(w).replace(old,new,1))
                replace2 = reverse_str(reverse_str(w).replace(new,old,1))

                if word_exists(replace1) :
                    print(f"Replace {old} with {new} -> {w} becomes {replace1} ")
                    good_words.append(replace1)
                if word_exists(replace2) :
                    print(f"Replace {new} with {old} -> {w} becomes {replace2} ")
                    good_words.append(replace2)


#In[]:


res0 = [p if p[-1].isalpha() else p[:-1] for p in good_words]
res0


#In[]:


res = list(filter(lambda x : len(x) > 2,set(res0)))
res.sort()
res


#In[]:


len(res)


#In[]:




