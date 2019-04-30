import argparse
import json
import sys
from fever.scorer import fever_score
from prettytable import PrettyTable
from analyse import print_confusion_mat, save_wrong_instances, save_simple_result, save_submission_file

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)

parser.add_argument("--predicted_evidence",type=str)
parser.add_argument("--actual",type=str)
parser.add_argument("--score_file",type=str)
parser.add_argument("--submission_file",type=str)
parser.add_argument("--err_analysis")
parser.add_argument("--test", action="store_true")
parser.add_argument("--predicted_labels_supplement")

args = parser.parse_args()

ids = []
predicted_labels =[]
predicted_evidence = []
actual = []

# make sure that using the corresponding data

actual_labels1 = list()
actual_labels2 = list()
actual_labels3 = list()


with open(args.predicted_labels,"r") as predictions_file:
    for line in predictions_file:
        predicted_labels.append(json.loads(line)["predicted"])


with open(args.predicted_evidence,"r") as predictions_file:
    for line in predictions_file:
        predicted_evidence.append(json.loads(line)["predicted_sentences"][:5])
        ids.append(json.loads(line)["id"])

predictions = []
for id, ev, label in zip(ids, predicted_evidence,predicted_labels):
    predictions.append({"id": id,"predicted_evidence":ev,"predicted_label":label})

save_submission_file(predictions, args.submission_file)
if args.test:
    print("abort scoring due to --test flag!")
    quit()


# with open(args.predicted_labels,"r") as predictions_file:
#     for line in predictions_file:
#         actual_labels1.append(json.loads(line)["actual"])

# with open(args.predicted_evidence,"r") as predictions_file:
#     for line in predictions_file:
#         actual_labels2.append(json.loads(line)["label"])

# with open(args.actual, "r") as actual_file:
#     for line in actual_file:
#         actual_labels3.append(json.loads(line)["label"])
#     for actual1, actual2, actual3 in zip(actual_labels1, actual_labels2, actual_labels3):
#         assert actual1 == actual2 == actual3, "{}, {}, {}".format(actual1, actual2, actual3)


with open(args.actual, "r") as actual_file:
    for line in actual_file:
        actual.append(json.loads(line))



score,acc,precision,recall,f1 = fever_score(predictions,actual)
save_simple_result(args.score_file, score, acc, precision, recall)
print_confusion_mat(predictions, actual)
if args.err_analysis:
    save_wrong_instances(args.actual, args.predicted_labels, args.predicted_evidence, args.err_analysis, args.predicted_labels_supplement)

tab = PrettyTable()
tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
tab.add_row((round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

print(tab)
