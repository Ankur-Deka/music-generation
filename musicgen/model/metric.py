import torch

def accuracy(Output, Target):
    output = Output['out']
    target = Target['out']
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0.0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(Output, Target, k=3):
    output = Output['out']
    target = Target['out']
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0.0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def top_k_note_accuracy(Output, Target, k=1):
    # Applicable for predictive model only
    gtnote = Target['nextnote']
    prednote = Output['nextnote']
    return top_k_acc({'out': prednote}, {'out': gtnote}, k)

def next_note_accuracy(output, target):
    return top_k_note_accuracy(output, target, 1)

def next_note_accuracy_5(output, target):
    return top_k_note_accuracy(output, target, 5)

def next_note_accuracy_10(output, target):
    return top_k_note_accuracy(output, target, 10)

def top_5_acc(output, target,):
    return top_k_acc(output, target, 5)

def top_10_acc(output, target,):
    return top_k_acc(output, target, 10)

