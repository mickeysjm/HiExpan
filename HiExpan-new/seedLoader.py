"""
__author__: Jiaming Shen, Ellen Wu
__description__: User input seed taxonomy used in HiExpan

Format of user supervision:
  [ParentEntityName, ParentLevel, ListOfChildrenEntityName]
Note:
  1. Except the ROOT name, all the other "ParentEntityName" must appear after its parent node appears
  (For example, "machine learning" must appear after "ROOT" (its parent)
  2. Don't add the leaf nodes in userInput
  (For example, "decision trees" is a leaf node, no need to include it in userInput)

"""


def load_seeds(corpusName):
    if corpusName == "wiki":
        userInput = [
            ["ROOT", -1, ["united states", "china", "canada"]],
            ["united states", 0, ["california", "illinois", "florida"]],
            ["china", 0, ["shandong", "zhejiang", "sichuan"]],
        ]
    elif corpusName == "dblp":
        userInput = [
            ["ROOT", -1, ["machine learning", "data mining", "natural language processing", "information retrieval", "wireless networks"]],
            ["data mining", 0, ["association rule mining", "text mining", "outlier detection"]],
            ["machine learning", 0, ["support vector machines", "decision trees", "neural networks"]],
            ["natural language processing", 0, ["named entity recognition", "information extraction", "machine translation"]],
        ]
    elif corpusName == "cvd":
        userInput = [
            ["ROOT", -1, ["cardiovascular abnormalities", "vascular diseases", "heart-disease"]],
            ["cardiovascular abnormalities", 0, ["turner syndrome", "tetralogy of fallot", "noonan syndrome"]],
            ["vascular diseases", 0, ["arteriovenous malformations", "high-blood pressure", "arterial occlusions"]],
            ["heart-disease", 0, ["aortic-valve stenosis", "cardiac arrests", "carcinoid heart disease"]],
        ]
    elif corpusName == "ql":
        userInput = [
            ["ROOT", -1, ["quantum algorithms", "quantum systems", "quantum theory"]],
            ["quantum algorithms", 0, ["quantum annealing", "quantum machine learning"]],
            ["quantum systems", 0, ["quantum computers", "quantum circuits"]],
            ["quantum theory", 0, ["quantum states", "hilbert-space"]],
        ]
    elif corpusName == "SignalProcessing":
        userInput = [
            ["ROOT", -1, ["acoustic_signal_processing", "adaptive_signal_processing", "digital_signal_processing", "image_processing"]],
            ["acoustic_signal_processing", 0, ["noise_reduction", "speech_processing", "speech_recognition"]],
            ["adaptive_signal_processing", 0, ["adaptive_filter"]],
            ["digital_signal_processing", 0, ["delta_modulation"]],
            ["image_processing", 0, ["image_processing_software"]],
        ]
    else:
        userInput = []

    return userInput