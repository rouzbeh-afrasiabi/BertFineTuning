# BertFineTuning
## (project in progress)
BERT Fine-tuning for Quora Question Pairs

Fine-tune BERT for duplicate detection with 9 lines of code


    from BertFineTuning.data_utils import *
    from BertFineTuning.model import *
    from pytorch_transformers.optimization import AdamW
    
    BFT=BertFineTuning()

    BFT.criterion=nn.CrossEntropyLoss()
    BFT.optimizer = AdamW(BFT.parameters_main)
    BFT.scheduler=torch.optim.lr_scheduler.MultiStepLR(BFT.optimizer, milestones=[])
    
    ml=MultiLoader()
    BFT.train(model_config,ml.bert_train_split,ml.bert_valid_split,epochs=100,print_every=100,validate_at_epoch=0)
Results for Training split:
<img src="/images/train.png"></img>

Results for validation split:
<img src="/images/test.png"></img>
