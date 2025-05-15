#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import transforms


# In[ ]:


import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


# In[ ]:


r_size = 256
input_size_1=1408
input_size_2=81
hidden_size=152
RESIZE=288
bert_emb_size=768
BATCH_SIZE=4


# In[ ]:


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# In[ ]:


from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


# In[ ]:


import time


# In[ ]:


#!pip install ipdb
#import ipdb


# In[ ]:


from transformers import BertTokenizer, BertModel


# In[ ]:


from pathlib import Path
import subprocess


# In[ ]:


import nltk
nltk.download('wordnet',quiet=True)
nltk.download('omw-1.4',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)

from nltk.corpus import wordnet
from nltk.corpus import stopwords

import spacy


# In[ ]:


nlp = spacy.load('it_core_news_sm')


# In[ ]:


KRAKEN_MODEL="10.5281/zenodo.10592716"
MODELNAME="catmus-print-fondue-large.mlmodel"
TRANSCRIPTION="krakenout.txt"
KRAKENGET_OUT="krakengetout"
MODELNAME_FILE = "modelname"


# In[ ]:


import os
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import random


# ### Data loader

# In[ ]:


# create a csv file with image_path and respective label
image_path_list = []
label_list = []
DATA_F = "classi"
for class_cat in os.listdir(DATA_F):
    print(class_cat)
    for image_object in os.listdir(f"{DATA_F}/{class_cat}"):
        if not image_object.endswith(".jpg"):
            continue
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


# In[ ]:


train_df.tail()


# In[ ]:


for cat in train_df.label.unique().tolist():
    elements = len(train_df.loc[train_df["label"] == cat])
    print(f"{cat} : {elements} items")


# In[ ]:


class CustomTrainingData(torch.utils.data.Dataset):

    bert_model = None
    bert_token = None

    def __init__(self, csv_df, class_list, transform=None, augment_factor=1):
        self.df = csv_df
        self.transform = transform
        self.class_list = class_list
        if augment_factor < 1:
            raise ValueError(f"You should increase dataset dimension, but Augmentation factor is {augment_factor}")
        self.augment_factor = int(augment_factor)

    def lemma_recursive(p):
        current = p
        update = ""
        while True:
            tokens = nlp(current)
            if len(tokens) > 0:
                update = tokens[0].__str__()

            if update == current:
                break
            current = update
        return update

    def reverse_str(s):
        return s[::-1]

    @classmethod   
    def word_exists(cls,p):
        return len(wordnet.lemmas(cls.lemma_recursive(p),lang="ita"))>0

    def clean_content(content : str) -> str:
        lines = content.split('\n')

        for i, l in enumerate(lines):
            lines[i] = ''.join([char if char.isalpha() else ' ' for char in lines[i]])

        for i, l in enumerate(lines):
            tokes = l.split(' ')
            tokes = list(filter(lambda x: any(list(map(lambda c : c.isalpha(),x))),tokes))
            lines[i] = ' '.join(tokes)


        for i, l in enumerate(lines):
            tokes = l.split(' ') # Now I'm sure there is no dumb token
            if i+1 < len(lines): # I can safely take a look to next line
                tokes.append(tokes[-1]+lines[i+1].split(' ')[0]) #Join last of this line with first of next line
                lines[i] = ' '.join(tokes)


        return ' '.join(lines)

    @classmethod
    def document_words_bert_embeddings_centroid(cls, src : str,bert_pretrained : str) -> torch.Tensor:
        if cls.bert_token is None:
            print("Init Bert tokenizer")
            cls.bert_token = BertTokenizer.from_pretrained(bert_pretrained)
        if cls.bert_model is None:
            print("Init Bert Model")
            cls.bert_model = BertModel.from_pretrained(bert_pretrained).to(device)
            for p in cls.bert_model.parameters():
                p.requires_grad = False
            cls.bert_model.eval()


        csvs = [x for x in Path(src).parents[0].glob('**/*') if x.is_file() and x.suffix == ".csv"]
        if len(csvs) == 0 :
            print("NO csv with transcription. Generating automatically")

            print("STEP 1 : Run OCR Model using Kraken. The model detects only printed and typewritten")

            TMP_FILE = Path(src).with_suffix(".kraken").resolve()#f"transcription_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if not TMP_FILE.exists():
                # kraken -d cuda:0 -i 0001_2_1881_015.jpg {TRANSCRIPTION} segment -bl ocr -m $(cat {MODELNAME_FILE})
                subprocess.run(["kraken", "-d", "cuda:0", "-i", src, TMP_FILE, "segment", "-bl", "ocr", "-m", MODELNAME]) 


            print("STEP 2 : Read file. Clean from special characters. Join words split due to newline")
            with open(TMP_FILE,"r") as fd:
                content1 = cls.clean_content(fd.read())
            compound = []
            for w in content1.split(' '):
                if any(map(lambda c : not c.isalnum(),w)):
                    letters = [c if c.isalpha() else ' ' for c in w]
                    #print(f" {list(w)} -> {letters}")
                    parts = ''.join(letters).split()
                    #print(parts)
                    if len(parts)>2:
                        continue
                    compound += parts
                    compound.append(''.join(parts))

            print("STEP 3 : Ask wordnet if knows about lemma equal to word (word exists). Also try replacing vocals with accents")
            good_words = []
            for w in content1.split(' ') + compound:

                if w in stopwords.words("italian"):
                    continue
                if cls.word_exists(w) :
                    good_words.append(w)
                else :
                    for old,new in [('a','à'),('e','è'),('e','é'),('i','ì'),('o','ò'),('u','ù')]:
                        if old in w or new in w :
                            replace1 = cls.reverse_str(cls.reverse_str(w).replace(old,new,1))
                            replace2 = cls.reverse_str(cls.reverse_str(w).replace(new,old,1))

                            if cls.word_exists(replace1) :
                                print(f"Replace {old} with {new} -> {w} becomes {replace1} ")
                                good_words.append(replace1)
                            if cls.word_exists(replace2) :
                                print(f"Replace {new} with {old} -> {w} becomes {replace2} ")
                                good_words.append(replace2)

            print(good_words)
            res0 = [p if p[-1].isalpha() else p[:-1] for p in good_words]
            print(res0)
            res1 = list(filter(lambda ww : len(ww) > 2,set(res0)))

            print(res1)

            with open(Path(src).with_suffix(".csv"),"w") as fd:
                for bow_el in res1:
                    fd.write(f"{bow_el}\n")
        #endIF bow file missing

        csvs = [x for x in Path(src).parents[0].glob('**/*') if x.is_file() and x.suffix == ".csv"] # again

        image_class_bow = []
        with open(csvs[0],"r") as fd: # CSVS[0] should not be out of bounds. If so, then just throw exception because i don't know what's going on
            for line in fd.read().split('\n'):
                if line != r"": # Last line is always empty due to how i print in file. That causes a tensor of Nan values to be added to set if line is not ignored
                    image_class_bow.append(line)

        #print(f"From file read bow {image_class_bow}")
        bow_droprate = 0.3

        while True: # Repeat the process if all words got dropped
            image_class_bow1 = list(filter(lambda x : random.random() < bow_droprate, image_class_bow)) # MASK words -> drop 1/3 of words
            if len(image_class_bow1) > 0 and len(image_class_bow1) < len(image_class_bow):
                break

        #print(f"After mask {image_class_bow1}")

        centroids = []
        for bag in [image_class_bow,image_class_bow1]:

            bert_enc = cls.bert_token.batch_encode_plus( bag,# List of input texts
                                                    padding=True,              # Pad to the maximum sequence length
                                                    truncation=True,           # Truncate to the maximum sequence length if necessary
                                                    return_tensors='pt',      # Return PyTorch tensors
                                                    add_special_tokens=False    # Add special tokens CLS and SEP
                                                ).to(device)

            with torch.no_grad():
                bert_out = cls.bert_model(bert_enc['input_ids'], attention_mask=bert_enc['attention_mask'])
                word_embeddings = bert_out.last_hidden_state

            #print("STEP 5 : Compute mean value of tokens in 'sentence' (the word) then mean value of all sentences")
            ocr_emb = word_embeddings.mean(dim=1).mean(dim=0)
            centroids.append(ocr_emb)

        if torch.equal(centroids[0],centroids[1]):
            print("After removing words centroid is the same")
        return torch.as_tensor(centroids[1])

    def __len__(self):
        return self.df.shape[0] * self.augment_factor

    def __getitem__(self, index):
        index = index % self.df.shape[0]
        source = Image.open(self.df.iloc[index].image_path).convert('RGB')
        label = self.class_list.index(self.df.iloc[index].label)

        if self.transform:
            image = self.transform(source)

        return image, label, CustomTrainingData.document_words_bert_embeddings_centroid(self.df.iloc[index].image_path,"bert-base-multilingual-uncased")


# In[ ]:


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
training_set = CustomTrainingData(train_df, train_df.label.unique().tolist(), transform, augment_factor=5)
validation_set = CustomTrainingData(test_df, test_df.label.unique().tolist(), transform_1, augment_factor=5)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

# Class labels
classes = tuple(train_df.label.unique().tolist())


# In[ ]:


classes


# In[ ]:


NUM_CLASSES = len(classes)


# ## Reference

# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


# In[ ]:


# EfficientNet_B<Y>, <Y> is one of the following: 0, 1, 2, 3

weights = EfficientNet_B2_Weights.DEFAULT
pretrained = efficientnet_b2(weights=weights).to(device)
# don't foget to call 'eval' to use the model for inference
pretrained.eval()


# #### Model

# In[ ]:


class DocClassifierTransfer(nn.Module):
    def __init__(self, num_classes, input_size_1, input_size_2, hidden_size):
        super(DocClassifierTransfer, self).__init__()
        self.num_classes = num_classes

        self.ln0 = nn.Linear(bert_emb_size,input_size_2)
        self.drop0 = nn.Dropout(p=0.2) #High dropout to avoid overfitting to single tensor per class


        self.ln1 = nn.Linear(input_size_2, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear((input_size_1+1)*hidden_size, self.num_classes)
        self.dropout = nn.Dropout(p=0.2)

        self.flat1 = nn.Flatten()

        weights_pre = EfficientNet_B2_Weights.DEFAULT
        self.pretrained = efficientnet_b2(weights=weights_pre).to(device)
        for p in self.pretrained.parameters():
            p.requires_grad = False

    def forward(self, x, centroid):
        #start = time.time()
        with torch.no_grad():
            x = self.pretrained.features(x)
            x = torch.reshape(x, (x.shape[0],x.shape[1], x.shape[2]*x.shape[3]))
        #print(f"EfficientNet B2 feature extraction time : {time.time() - start}")
        centroid = self.ln0(centroid) # Now shape should be batch_size x input_size_2
        centroid = self.drop0(centroid)
        centroid = centroid.unsqueeze(1)
        x = torch.cat((x,centroid),dim=1)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flat1(x)
        x = self.ln2(x)
        return x

model = DocClassifierTransfer(NUM_CLASSES, input_size_1, input_size_2, hidden_size).to(device)


# In[ ]:


sample_data = next(iter(training_loader))
for i in range(0,len(sample_data)):
    if isinstance(sample_data[i],torch.Tensor):
        print(sample_data[i].shape)
    else:
        print(sample_data[i])


# In[ ]:


summary(model=model, input_data=(sample_data[0].to(device),sample_data[2].to(device)), col_names=['input_size', 'output_size', 'num_params', 'trainable'])


# ## Classic pytorch training loop

# ### Optimizer and Loss function

# In[ ]:


#loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# ### Per batch

# In[ ]:


def train_one_epoch(epoch_index):#, tb_writer):
    running_loss = 0.
    last_loss = 0.


    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):

        # Every data instance is an input + label pair
        inputs, labels, centroid = data
        inputs, labels, centroid = inputs.to(device), labels.to(device), centroid.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs,centroid)

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
        every_n = min(len(training_loader.dataset)/(10*BATCH_SIZE),1000)
        every_n = int(every_n)

        if i % every_n == (every_n - 1 ):
            last_loss = running_loss / (every_n) # loss per batch
            print(f'{(int(i/every_n)+1):02d}) batch {i + 1} loss: {last_loss:.4f}')
            #tb_x = epoch_index * len(training_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.


    return last_loss


# ### Per Epoch

# In[31]:


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH{}:'.format(epoch_number + 1))

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
            vinputs, vlabels, vcentr = vdata
            vinputs, vlabels, vcentr = vinputs.to(device), vlabels.to(device), vcentr.to(device)
            voutputs = model(vinputs,vcentr)
            vloss = loss_fn(voutputs, vlabels)

            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss:.4f} valid {avg_vloss:.4f}')

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


# --------
# 
# Both train loss and validation loss should be less than `0.2`
# 
# [Reference : Interpretation of Cross-Entropy values](https://wiki.cloudfactory.com/docs/mp-wiki/loss/cross-entropy-loss#:~:text=Cross%2DEntropy%20%3C%200.05%3A%20On,%2DEntropy%20%3E%201.00%3A%20Terrible.)
# 
# --------

# In[ ]:


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

