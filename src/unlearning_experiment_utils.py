
import torch 
from torch import optim, nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy


from kegnet.classifier.models import ALLCNN
from kegnet.utils import data, utils
from kegnet.classifier.train import eval_classifier, predict_labels

from kegnet.generator.utils import sample_kegnet_data

from sklearn.linear_model import LogisticRegression


batch_size = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    # data_loader = torch.utils.data.DataLoader(
    #     data_loader.dataset, batch_size=1, shuffle=False
    # )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            # batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            data = data.cuda()
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r

def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


def get_retain_dataloader_and_forget_dataloader( train_dataset, target_class_id):
    
    # train_dataset_x, train_dataset_y = d.trn_data

    # split by class
    splitted_dataset = {}

    for index, _ in enumerate( train_dataset ):
        
        if ( train_dataset[index][1] in  splitted_dataset  ):
            splitted_dataset[ train_dataset[index][1]  ]['x'].append( train_dataset[index][0].unsqueeze(0)  )
            splitted_dataset[ train_dataset[index][1]  ]['y'].append( train_dataset[index][1]  )
        else:
            splitted_dataset[ train_dataset[index][1]  ] = {
                "x": [train_dataset[index][0].unsqueeze(0) ],
                "y": [train_dataset[index][1]],
            }

    # get dr and df lists
    Dr = {"x":[], 'y':[]}
    Df = {"x":[], 'y':[]}
    for class_id in splitted_dataset:
        if class_id == target_class_id:
            Df["x"].extend( splitted_dataset[class_id]["x"] )
            Df["y"].extend( splitted_dataset[class_id]["y"] )
        else:
            Dr["x"].extend( splitted_dataset[class_id]["x"] )
            Dr["y"].extend( splitted_dataset[class_id]["y"] )
            
    # create a dataloader 

    real_dr_dataloader = DataLoader(TensorDataset( torch.vstack(Dr["x"]) , torch.tensor(Dr["y"] ) ), batch_size)
    real_df_dataloader = DataLoader(TensorDataset( torch.vstack(Df["x"]) , torch.tensor(Df["y"] )  ), batch_size)

    return real_dr_dataloader, real_df_dataloader

def print_DfAcc_DrAcc(conf_mat, target_class_id):
    acc = []
    df_Acc = []
    dr_ACC = []
    for i in range(10):
        if ( i == target_class_id):
            df_Acc.append( conf_mat[i][i] )
        else:
            dr_ACC.append(conf_mat[i][i] )
        
        acc.append(conf_mat[i][i])
    
    print (acc)
    print (f"Df Acc =  {np.mean(df_Acc)}",
           f"Dr Acc =  {np.mean(dr_ACC)}")
    

def NegGrad( classifier, forget_dataloader, retain_dataloader, optimizer, train_with_retain = True ):
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    for images, labels in forget_dataloader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = -loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if (train_with_retain):
        for images, labels in retain_dataloader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return classifier

def our_method( model, forget_dataloader, retain_dataloader, learning_rate, train_with_retain=True ):
    class DistillKL(nn.Module):
        def __init__(self, T):
            super(DistillKL, self).__init__()
            self.T = T

        def forward(self, y_s, y_t):
            p_s = F.log_softmax(y_s/self.T, dim=1)
            p_t = F.softmax(y_t/self.T, dim=1)
            loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
            return loss
        
    criterion = DistillKL(4.0)
    
    ref_model = deepcopy(model)
    ref_model.eval()

    model.train()
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    print( "forget  step..." )
    for batch in  forget_dataloader :
        images, labels = batch
        random_input = torch.rand( images.shape, device=DEVICE )
        
        out_r = ref_model(random_input)   
        out = model(images)                  
        
        loss = criterion(out, out_r)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if train_with_retain:
        print( "retain  step..." )
        for remember_batch in retain_dataloader:
            images, labels = remember_batch
            out_r = ref_model(images)   
            out = model(images)                  
            loss = criterion(out, out_r)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model
        

