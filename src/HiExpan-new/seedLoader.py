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
            ["ROOT", -1, ["united_states", "china", "canada"]],
            ["united_states", 0, ["california", "illinois", "florida"]],
            ["china", 0, ["shandong", "zhejiang", "sichuan"]],
        ]
    elif corpusName == "dblp":
        userInput = [
            ["ROOT", -1, ["machine_learning", "data_mining", "natural_language_processing", "information_retrieval", "wireless_networks"]],
            ["data_mining", 0, ["association_rule_mining", "text_mining", "outlier_detection"]],
            ["machine_learning", 0, ["support_vector_machines", "decision_trees", "neural_networks"]],
            ["natural_language_processing", 0, ["named_entity_recognition", "information_extraction", "machine_translation"]],
        ]
    elif corpusName == "cvd":
        userInput = [
            ["ROOT", -1, ["cardiovascular_abnormalities", "vascular_diseases", "heart-disease"]],
            ["cardiovascular_abnormalities", 0, ["turner_syndrome", "tetralogy_of_fallot", "noonan_syndrome"]],
            ["vascular_diseases", 0, ["arteriovenous_malformations", "high-blood_pressure", "arterial_occlusions"]],
            ["heart-disease", 0, ["aortic-valve_stenosis", "cardiac_arrests", "carcinoid_heart_disease"]],
        ]
    elif corpusName == "ql":
        userInput = [
            ["ROOT", -1, ["quantum_algorithms", "quantum_systems", "quantum_theory"]],
            ["quantum_algorithms", 0, ["quantum_annealing", "quantum_machine_learning"]],
            ["quantum_systems", 0, ["quantum_computers", "quantum_circuits"]],
            ["quantum_theory", 0, ["quantum_states", "hilbert-space"]],
        ]
    elif corpusName == "SignalProcessing":
        userInput = [
            ["ROOT", -1, ["acoustic_signal_processing", "adaptive_signal_processing", "digital_signal_processing", "image_processing"]],
            ["acoustic_signal_processing", 0, ["noise_reduction", "speech_processing", "speech_recognition"]],
            ["adaptive_signal_processing", 0, ["adaptive_filter"]],
            ["digital_signal_processing", 0, ["delta_modulation"]],
            ["image_processing", 0, ["image_processing_software"]],
        ]
    elif corpusName == "sample_dataset":
        userInput = [
            ["ROOT", -1, ["machine_learning", "data_mining", "database"]]
        ]
    else:
        userInput = []

    return userInput
