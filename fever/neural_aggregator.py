import sys
from collections import Counter
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from fever_io import read_jsonl, save_jsonl

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

class Net(nn.Module):
    def __init__(self, layers=[15, 10, 5]):
        super(Net, self).__init__()
        assert len(layers) == 3, "currently, only supports 3 layer MLP"
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PredictedLabelsDataset(Dataset):
    """Predicted Labels dataset."""

    def __init__(self, jsonl_file, n_sentences=5, sampling=False, use_ev_scores=False, test=False):
        """
        """
        instances = read_jsonl(jsonl_file)
        if sampling:
            instances = sample(instances)

        self.instances = instances
        self.n_sentences = n_sentences
        self.test = test
        self.use_ev_scores = use_ev_scores

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        if self.test:
            label = create_target("DUMMY LABEL")
        else:
            label = create_target(self.instances[idx]["label"])

        if not self.instances[idx]["scores"]:
            pass
            #print("missing", idx)
            # print(self.instances[idx])
            # import ipdb
            # ipdb.set_trace()
        if self.use_ev_scores:
            input = create_input3(  # rather than create_input2
                self.instances[idx]["predicted_labels"],
                self.instances[idx]["scores"],
                self.instances[idx]["ev_scores"],
                n_sentences=self.n_sentences)
        else:
            input = create_input(
                self.instances[idx]["predicted_labels"],
                self.instances[idx]["scores"],
                n_sentences=self.n_sentences)

        return (label, input)
        # return (self.instances[idx]["label"], self.instances[idx]["predicted_labels"]) #, self.instances[idx]["scores"])


def sample(train_set):
    print("performing sampling...")
    sampled_instances = list()
    label2freq = Counter((instance["label"] for instance in train_set))
    print("label2freq:", label2freq)
    min_freq = min(label2freq.values())
    counter_dict = dict()
    for instance in train_set:
        label = instance["label"]
        if label not in counter_dict:
            counter_dict[label] = 1
        elif counter_dict[label] < min_freq:
            counter_dict[label] += 1
            sampled_instances.append(instance)

    assert all(count == min_freq for count in counter_dict.values())
    return sampled_instances

zero_plus_eye = np.vstack([np.eye(3), np.zeros((1, 3))])
zero_pad_idx = 3


def create_input(predicted_labels, scores, n_sentences):
    pred_labels = [label2idx[pred_label] for pred_label in predicted_labels]
    scores = scores.copy()

    # fill zero if number of predicted_labels (it corresponds to the predicted evidences) are less than n_sentences
    if len(pred_labels) < n_sentences:
        n_fillup = n_sentences - len(pred_labels)
        pred_labels += [zero_pad_idx] * n_fillup
        scores += [0.] * n_fillup

    one_hot = zero_plus_eye[
        pred_labels, :]  # equivalent to embedding_lookup(pred_labels, eye)

    # np_out = np.mean(np.multiply(one_hot, np.expand_dims(scores, axis=1)), axis=0)
    np_out = np.reshape(
        np.multiply(one_hot, np.expand_dims(scores, axis=1)), (-1))
    return np_out


def create_input3(predicted_labels, scores, sentence_scores, n_sentences):
    """
    4 features for each predicted evidence: the 3 RTE class probabilities, and the IR score.
    All 3 class probabilities are given.
    """
    if len(predicted_labels) == 1:
        predicted_labels = predicted_labels[0]

    assert len(predicted_labels) == len(scores)

    if len(sentence_scores) == 0:
        return np.zeros([4*n_sentences])

    # loop over the predicted evidences
    features_per_predicted_evidence = []
    for per_evidence_labels, per_evidence_scores, ir_evidence_scores in \
                        list(zip(predicted_labels, scores, sentence_scores))[:n_sentences]:

        if not per_evidence_scores: # sometimes this is empty.
            new_features = [0.0, 0.0, 0.0, 0.0]
        else:
            # default case
            new_features = per_evidence_scores + [ir_evidence_scores[2]]

        # 4 new features
        features_per_predicted_evidence.extend(new_features)

    for missing_evidence in range(n_sentences - len(sentence_scores)):
        features_per_predicted_evidence.extend([0.0, 0.0, 0.0, 0.0])

    np_out = np.array(features_per_predicted_evidence)

    if np_out.shape != (4*n_sentences,):
        import ipdb
        ipdb.set_trace()
    assert np_out.shape == (4*n_sentences,), (np_out, len(predicted_labels), len(scores), len(sentence_scores))
    return np_out


def create_input2(predicted_labels, scores, sentence_scores, n_sentences):
    """
    4 features for each predicted evidence: the 3 RTE class probabilities, and the IR score.
    However, the non-winning class probabilies are set to 0.
    """
    pred_labels = [label2idx[pred_label] for pred_label in predicted_labels]
    scores = scores.copy()
    ev_scores = [score for _, _, score in sentence_scores][:len(scores)]
    if len(ev_scores) < n_sentences:
        n_fillup = n_sentences - len(ev_scores)
        ev_scores += [0.] * n_fillup


    # fill zero if number of predicted_labels (it corresponds to the predicted evidences) are less than n_sentences
    if len(pred_labels) < n_sentences:
        n_fillup = n_sentences - len(pred_labels)
        pred_labels += [zero_pad_idx] * n_fillup
        scores += [0.] * n_fillup

    one_hot = zero_plus_eye[
        pred_labels, :]  # equivalent to embedding_lookup(pred_labels, eye)

    # np_out = np.mean(np.multiply(one_hot, np.expand_dims(scores, axis=1)), axis=0)
    np_rtes = np.reshape(
        np.multiply(one_hot, np.expand_dims(scores, axis=1)), (-1))
    np_irs = np.array(ev_scores)
    np_out = np.concatenate([np_rtes, np_irs])
    return np_out


def create_target(label):
    label_idx = label2idx[label]
    return label_idx


# use primodial aggregate preds function
def aggregate_preds(predictions):
    vote = dict()
    for rank, pred in enumerate(predictions):
        if pred not in vote:
            vote[pred] = 1
        else:
            vote[pred] += 1

    if supports in vote and refutes in vote:
        for pred in predictions:
            if pred in [supports, refutes]:
                final_verdict = pred
                break
    elif supports in vote:
        final_verdict = supports
    elif refutes in vote:
        final_verdict = refutes
    else:
        final_verdict = nei
    return final_verdict


def simple_test(dev_dataloader):
    # test for prediction
    neural_hit = 0
    heuristic_hit = 0
    with torch.no_grad():
        for i, (target, input) in enumerate(dev_dataloader):
            neural_pred = net(input.float())
            _, pred_labels = torch.max(neural_pred, 1)
            neural_hit += torch.sum(pred_labels == target)
    performance = int(neural_hit) / len(dev_dataloader.dataset)
    print("neural:", performance)
    return performance
    # for i, instance in enumerate(dev_dataloader.dataset.instances):
    #     heuristic_pred_label = aggregate_preds(instance["predicted_labels"])
    #     if heuristic_pred_label == instance["label"]:
    #         heuristic_hit += 1
    # print("heuristic:", heuristic_hit / len(dev_dataloader.dataset.instances))


def predict(test_dataloader):
    results = list()
    with torch.no_grad():
        for i, (labels, input) in enumerate(test_dataloader):
            neural_preds = net(input.float())
            _, pred_labels = torch.max(neural_preds, 1)

            for label, neural_pred in zip(labels, pred_labels):
                results.append({
                    "actual": idx2label[int(label)],
                    "predicted": idx2label[int(neural_pred)]
                })

    return results


label2idx = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2, "DUMMY LABEL": 3}
idx2label = {idx: label for label, idx in label2idx.items()}
supports = idx2label[0]
refutes = idx2label[1]
nei = idx2label[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--n_sentences", default=5, type=int)
    parser.add_argument("--predicted_labels", required=True)
    parser.add_argument("--test_predicted_labels", required=True)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--ev_scores", action="store_true")
    parser.add_argument("--l2", default=0.0, type=float, required=False)
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        help=
        "<required> specify each width of layers (currently, should be 3 layers)"
    )
    args = parser.parse_args()

    # data: prepend_title_linum
    print(args)
    hyperparameter2performance = dict()
    for n_sentences in [5]:#range(1, 16):
        print("=========== n_sentences {}============".format(str(n_sentences)))
        args.n_sentences = n_sentences
        # number of inputs will be 4 times number of evidence sentences.
        args.layers[0] = args.n_sentences * 4


        train_set = PredictedLabelsDataset(args.train, args.n_sentences, sampling=args.sampling, use_ev_scores=args.ev_scores)
        dev_set = PredictedLabelsDataset(args.dev, args.n_sentences, use_ev_scores=args.ev_scores)
        test_set = PredictedLabelsDataset(args.test, args.n_sentences, use_ev_scores=args.ev_scores, test=True)
        train_dataloader = DataLoader(
            train_set, batch_size=64, shuffle=True, num_workers=0)
        dev_dataloader = DataLoader(
            dev_set, batch_size=64, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=0)


        net = Net(layers=[int(width) for width in args.layers])
        print("----Neural Aggregator Architecture----")

        print(net)

        class_weights = [1.0, 1.0, 1.0]
        if 1:   # properly set class weights
            label2freq = Counter((instance["label"] for instance in train_set.instances))
            class_weights = [0, 0, 0]
            total = sum(label2freq.values())
            for label in label2freq:
                class_weights[ label2idx[label] ] = 1.0/(label2freq[label])*total
            print(label2freq)
            print("Class Weights:", class_weights)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        #criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(net.parameters(), weight_decay=args.l2/2.0)
        optimizer = optim.Adam(net.parameters())
        dev_results_throughout_training = []
        for epoch in range(args.epochs):  # loop over the dataset multiple times
            print("epoch:", epoch)
            running_loss = 0.0

            for i, (labels, inputs) in enumerate(train_dataloader):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:  # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                                    running_loss / 1000))
                    running_loss = 0.0
            # monitor dev loss throughout training
            dev_results_throughout_training.append(simple_test(dev_dataloader))

        print('Finished Training')

        print("dev set:")
        performance = simple_test(dev_dataloader)
        #hyperparameter2performance[n_sentences] = performance
        hyperparameter2performance[n_sentences] = max(dev_results_throughout_training)

    for k, v in sorted(hyperparameter2performance.items()):
        print(v)


    dev_results = predict(dev_dataloader)
    test_results = predict(test_dataloader)
    save_jsonl(dev_results, args.predicted_labels)
    save_jsonl(test_results, args.test_predicted_labels)