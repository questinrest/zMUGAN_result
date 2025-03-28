import torch
import torch.nn as nn
from unlearning_experiment_utils import *
from kegnet.generator.utils import sample_kegnet_data
from kegnet.classifier.models import lenet,resnet,VGG16,ALLCNN
from kegnet.utils import data, utils
from kegnet.classifier.train import eval_classifier, predict_labels
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression





target_class_id = 3

dataset = "svhn"
model_name = "allcnn"
batch_size = 32

path = "/kaggle/working/zMUGAN_result/svhn_allcnn.pth" ##path to teacher

d = data.to_dataset(dataset)
if model_name == "vgg16":
    classifier = VGG16.VGG16()
else:
    classifier = AllCNN.AllCNN()
checkpoint = torch.load(path, map_location=DEVICE)
model_state = checkpoint.get('model_state', None)
classifier.load_state_dict(model_state)
classifier = classifier.cuda()

trn_data, val_data, test_data = d.to_loaders(batch_size)
loss_func = nn.CrossEntropyLoss().to(DEVICE)

# calculate per class accuracy
test_loss, test_acc, true_y, pred_y = eval_classifier(classifier, test_data, loss_func, return_true_and_pred=True)

# load generators 
generators_paths = [ 
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-0/generator-100.pth.tar", 
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-0/generator-200.pth.tar", 
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-1/generator-100.pth.tar", 
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-1/generator-200.pth.tar", 
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-2/generator-100.pth.tar",
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-2/generator-200.pth.tar",
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-3/generator-100.pth.tar",
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-3/generator-200.pth.tar",
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-4/generator-100.pth.tar",
                    "/kaggle/working/zMUGAN_result/src/path_o/generator-4/generator-200.pth.tar"
                    ]



for num_data_ in [5000]:
    
    print ("model loaded")
    if model_name == "vgg16":
        classifier = VGG16.VGG16()
    else:
        classifier = AllCNN.AllCNN()
        
    checkpoint = torch.load(path, map_location=DEVICE)
    model_state = checkpoint.get('model_state', None)
    classifier.load_state_dict(model_state)
    classifier = classifier.cuda()
        
    # generate dataset
    generated_images = sample_kegnet_data(dataset=dataset, device=DEVICE, generators=generators_paths,num_data=num_data_)
    labels = predict_labels(classifier, generated_images)




    images_labels_pairs = zip( generated_images, labels )

    forget_dataset_images = []
    forget_dataset_labels = []

    retain_dataset_images = []
    retain_dataset_labels = []

    class_count = {}

    # print ("before unlearning")
    # print_DfAcc_DrAcc(conf_mat, target_class_id)

    for index, pair in enumerate(images_labels_pairs):
        image, label_prop = pair
        
        predicted_label = label_prop.argmax(dim=0)

        if ( predicted_label.item() in class_count ):
            class_count[  predicted_label.item() ] +=1 
        else :
            class_count[  predicted_label.item() ] = 1 

        if ( predicted_label.item() == target_class_id ):
            forget_dataset_images.append( image.unsqueeze(0) )
            forget_dataset_labels.append( label_prop )
        else:
            retain_dataset_images.append( image.unsqueeze(0) )
            retain_dataset_labels.append( label_prop )

    print ( class_count )    

    forget_dataset_images, forget_dataset_labels = torch.vstack(forget_dataset_images), torch.vstack(forget_dataset_labels)
    retain_dataset_images, retain_dataset_labels = torch.vstack(retain_dataset_images), torch.vstack(retain_dataset_labels)

    print ("df_Size = ", len(forget_dataset_images)  )
    print ("dr_Size = ", len(retain_dataset_images)  )

    forget_data_loader = DataLoader(TensorDataset(forget_dataset_images, forget_dataset_labels), batch_size)
    retain_data_loader = DataLoader(TensorDataset(retain_dataset_images, retain_dataset_labels), batch_size)


    train_dataset, test_dataset = data.to_dataset(dataset).get_dataset()
    real_dr_dataloader, real_df_dataloader = get_retain_dataloader_and_forget_dataloader(train_dataset, target_class_id )

    lr = 1e-5
    # for lr in [ 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1   ]:
        # for target_class_id in range (10):
    print ( f"target_class {target_class_id}" )
    
    print ( f"learning rate = {lr}" )

    classifier.train()
    print ("unlearning started")
    classifier = our_method( classifier, forget_data_loader, retain_data_loader,  lr )
    print ("unlearning done")

    # calculate per class accuracy again
    test_loss, test_acc, true_y, pred_y = eval_classifier(classifier, test_data, loss_func, return_true_and_pred=True)
    conf_mat = confusion_matrix(true_y, pred_y, normalize="true")

    print ("after unlearning")
    print_DfAcc_DrAcc(conf_mat, target_class_id)

    # call the function
    classifier.eval()
    mia_score = get_membership_attack_prob(real_dr_dataloader, real_df_dataloader, test_data, classifier)

    print ("mia_score", mia_score)

