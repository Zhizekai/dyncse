print("开始运行")
# from transformers import BertTokenizer
# from torch.utils.data import DataLoader
from datasets import load_from_disk #, load_dataset, Dataset
# import pandas as pd
from torch.utils.data import DataLoader
import torch
# from mocose import *
# from transformers import BertConfig
# from mocose_tools import MoCoSETrainer
# from transformers.trainer import TrainingArguments
import argparse
import torch.nn.functional as F
from transformers import (
    CONFIG_MAPPING, 
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from tqdm import tqdm

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 64
# train_dataset = load_from_disk("F:\\Models\\temp\\wiki_for_sts_32")
# train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last = True)


def align_loss(x, y, alpha=2):    
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def get_pair_emb(model, input_ids, attention_mask,token_type_ids):    
    outputs = model(input_ids = input_ids.cuda(),attention_mask=attention_mask.cuda(),token_type_ids=token_type_ids.cuda())
   
    # outputs.last_hidden_state.shape [128, 32, 768]
    pooler_output = outputs.last_hidden_state[:,0]     
    #pooler_output = outputs.pooler_output
    z1, z2 = pooler_output[:batch_size], pooler_output[batch_size:]
    # z1 [64, 768]
    return z1,z2

def get_align(model, dataloader):
    align_all = []
    with torch.no_grad():        
        for data in tqdm(dataloader, desc="正在测试 align 中，请等待"):
            input_ids = torch.cat((data['input_ids'][:,0],data['input_ids'][:,1]))
            attention_mask = torch.cat((data['attention_mask'][:,0],data['attention_mask'][:,1]))
            token_type_ids = torch.cat((data['token_type_ids'][:,0],data['token_type_ids'][:,1]))

            z1,z2 = get_pair_emb(model, input_ids, attention_mask, token_type_ids)  
            z1 = F.normalize(z1,dim=1)
            z2 = F.normalize(z2,dim=1)
            align_all.append(align_loss(z1, z2))
            
    return align_all
    
def get_unif(model, dataloader):
    unif_all = []
    with torch.no_grad():        
        for data in tqdm(dataloader, desc="正在测试 uniform 中，请等待"):
            # [64, 2, 32]
            input_ids = torch.cat((data['input_ids'][:,0],data['input_ids'][:,1]))
            attention_mask = torch.cat((data['attention_mask'][:,0],data['attention_mask'][:,1]))
            token_type_ids = torch.cat((data['token_type_ids'][:,0],data['token_type_ids'][:,1]))

            z1,z2 = get_pair_emb(model, input_ids, attention_mask, token_type_ids)        
            z1 = F.normalize(z1,dim=1)  # 默认就是使用l2 正则化
            z2 = F.normalize(z2,dim=1)
            z = torch.cat((z1,z2))
            unif_all.append(uniform_loss(z1, t=2))
            
    return unif_all

print("开始加载数据")
pos_dataset = load_from_disk("/mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/mocose-main/data/uniform_align_data/stsb_pos")
pos_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
pos_loader = DataLoader(pos_dataset, batch_size = batch_size, drop_last = True)

all_dataset = load_from_disk("/mnt/nfs-storage-pvc-n26-20241218/rizejin/zzk/dyncse/mocose-main/data/uniform_align_data/stsb_all")
all_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
all_loader = DataLoader(all_dataset, batch_size = batch_size, drop_last = True)
print("加载数据完成")

# trainer.train()  # 训练模型

# 加载模型
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, help="Transformers' model name or path")

args = parser.parse_args()

# Load transformers' model checkpoint
print("开始加载模型")
print(args.model_name_or_path)

# bert
model_ = BertModel.from_pretrained(args.model_name_or_path)

# roberta
# model_ = RobertaModel.from_pretrained(args.model_name_or_path)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ = model_.to(device)

# tokenizer = BertTokenizer(vocab_file=os.path.join(args.model_name_or_path, "vocab.txt"))
# tokenizer = RobertaTokenizer(vocab_file=os.path.join(args.model_name_or_path, "vocab.json"), merges_file=os.path.join(args.model_name_or_path, "merges.txt"))

print("已加载模型")

align_all = get_align(model_, all_loader)
align_pos = get_align(model_, pos_loader)
uniform_all = get_unif(model_, all_loader)
uniform_pos = get_unif(model_, pos_loader)

all_loader_align = sum(align_all)/len(align_all)
pos_loader_align = sum(align_pos)/len(align_pos)
all_loader_uniformity = sum(uniform_all)/len(uniform_all)
pos_loader_uniformity = sum(uniform_pos)/len(uniform_pos)

print("--------------alignment & uniformity-----------------")
print("all_loader_align",all_loader_align)
print("pos_loader_align",pos_loader_align)
print("all_loader_uniformity",all_loader_uniformity)
print("pos_loader_uniformity",pos_loader_uniformity)