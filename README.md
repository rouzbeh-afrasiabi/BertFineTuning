# BertFineTuning
## (project in progress)


Fine-tune BERT for duplicate detection with 9 lines of code. The code can be modified for fine-tuning BERT on custom datasets.


    from BertFineTuning.data_utils import *
    from BertFineTuning.model import *
    from pytorch_transformers.optimization import AdamW
    
    BFT=BertFineTuning()

    BFT.criterion=nn.CrossEntropyLoss()
    BFT.optimizer = AdamW(BFT.parameters_main)
    BFT.scheduler=torch.optim.lr_scheduler.MultiStepLR(BFT.optimizer, milestones=[])
    
    ml=MultiLoader()
    BFT.train(model_config,ml.bert_train_split,ml.bert_valid_split,epochs=100,print_every=100,validate_at_epoch=0)
 
 
Loss for Training and Test splits:<br>
<p align="left">
<img src="/images/loss.png"></img>
</p>
 
Accuracy and F1 for Training split:<br>
<p align="left">
<img src="/images/train.png"></img>
</p>

Accuracy and F1 for validation split:<br>
<p align="left">
<img src="/images/test.png"></img>
</p>

## Requirements
        pycm==2.2
        pandas==0.25.1
        torch==1.2.0
        spacy==2.1.4
        requests==2.22.0
        pytorch_transformers==1.0.0
        numpy==1.17.2
        scikit_learn==0.21.3
