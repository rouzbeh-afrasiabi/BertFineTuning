from BertFineTuning.data_utils import *
from BertFineTuning.BertFineTuning import *
from pytorch_transformers.optimization import AdamW


def train():
    
    BFT=BertFineTuning()

    BFT.criterion=nn.CrossEntropyLoss()
    BFT.optimizer = AdamW(BFT.parameters_main)
    BFT.scheduler=torch.optim.lr_scheduler.MultiStepLR(BFT.optimizer, milestones=[])

    ml=MultiLoader()

    BFT.train(model_config,ml.bert_train_split,ml.bert_valid_split,epochs=100,print_every=100,validate_at_epoch=0)
    print("Training in progress ...")
if (__name__ == "__main__"):
    train()