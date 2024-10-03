from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, Accuracy

class f1_metrics:
    def __init__(self, num_classes, partition, set_name):
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.accuracy_2 = Accuracy(task='multiclass', num_classes=num_classes, top_k=2)
        self.f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro')
        self.precision_macro = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.precision_micro = MulticlassPrecision(num_classes=num_classes, average='micro')
        self.recall_macro = MulticlassRecall(num_classes=num_classes, average='macro')
        self.recall_micro = MulticlassRecall(num_classes=num_classes, average='micro')
        self.set_name = set_name
        self.partition = partition
    
    def update(self, pred, target):
        self.accuracy.update(pred, target)
        self.accuracy_2.update(pred, target)
        self.f1_macro.update(pred, target)
        self.f1_micro.update(pred, target)
        self.precision_macro.update(pred, target)
        self.precision_micro.update(pred, target)
        self.recall_macro.update(pred, target)
        self.recall_micro.update(pred, target)

    def compute(self):
        acc = self.accuracy.compute()
        acc_2 = self.accuracy_2.compute()
        f1_macro = self.f1_macro.compute()
        f1_micro = self.f1_micro.compute()
        precision_macro = self.precision_macro.compute()
        precision_micro = self.precision_micro.compute()
        recall_macro = self.recall_macro.compute()
        recall_micro = self.recall_micro.compute()
        return (acc, acc_2, f1_macro, f1_micro, precision_macro, 
                precision_micro, recall_macro, recall_micro)

    def compute_nonzero_f1(self):
        # Calculate the F1 score for each class and then filter out the zero values
        f1_per_class = self.f1_macro.compute(per_class=True)
        nonzero_f1 = f1_per_class[f1_per_class != 0]
        return nonzero_f1

    def show(self, _class=None):
        (accuracy, 
        accuracy_2,
        f1_macro, 
        f1_micro, 
        precision_macro, 
        precision_micro, 
        recall_macro, 
        recall_micro) = self.compute()
        print(self.set_name + ' accuracy: ', accuracy)
        print(self.set_name + ' accuracy@2: ', accuracy_2)
        print('Macro ' + self.partition + ' ' + self.set_name + ' f1: ', f1_macro)
        print('Micro ' + self.partition + ' ' + self.set_name + ' f1: ', f1_micro)
        print('Macro ' + self.partition + ' ' + self.set_name + ' precision: ', precision_macro)
        print('Micro ' + self.partition + ' ' + self.set_name + ' precision: ', precision_micro)
        print('Macro ' + self.partition + ' ' + self.set_name + ' recall: ', recall_macro)
        print('Micro ' + self.partition + ' ' + self.set_name + ' recall: ', recall_micro)
        
        if _class is not None:
            print('Macro ' + self.partition + ' ' + self.set_name + ' f1 for class ' + str(_class), f1_macro[_class].item())
            print('Micro ' + self.partition + ' ' + self.set_name + ' f1 for class ' + str(_class), f1_micro[_class])
            print('Macro ' + self.partition + ' ' + self.set_name + ' precision for class ' + str(_class), precision_macro[_class])
            print('Micro ' + self.partition + ' ' + self.set_name + ' precision for class ' + str(_class), precision_micro[_class])
            print('Macro ' + self.partition + ' ' + self.set_name + ' recall for class  ' + str(_class), recall_macro[_class])
            print('Micro ' + self.partition + ' ' + self.set_name + ' recall for class ' + str(_class), recall_micro[_class])

        # Display nonzero F1 scores
        nonzero_f1 = self.compute_nonzero_f1()
        print('Non-zero F1 scores: ', nonzero_f1)

        return f1_macro, f1_micro, nonzero_f1
