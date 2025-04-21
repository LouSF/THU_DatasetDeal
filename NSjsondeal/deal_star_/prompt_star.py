prompt_template_input = {
    "Sequence": {
        "T1": r"^Which object did the person (.*?) after they (.*?) the (.*?)\?",
        "T2": r"^Which object did the person (.*?) before they (.*?) the (.*?)\?",
        "T3": r"^What happened after the person (.*?) the (.*?)\?",
        "T4": r"^What happened before the person (.*?) the (.*?)\?",
        "T5": r"^What did the person do to the (.*?) after (.*?) the (.*?)\?",
        "T6": r"^What did the person do to the (.*?) before (.*?) the (.*?)\?",
    },
    "Prediction": {
        "T1": r"^What will the person do next?",
        "T2": r"^What will the person do next with the (.*?)\?",
        "T3": r"^Which object would the person (.*?) next\?",
        "T4": r"^Which object would the person (.*?) next after they (.*?) the (.*?)\?",
    }
}

prompt_target = {
    "Sequence": [
        [
            ("During {} seconds to {} seconds, the person did two things. Please list them sequentially.", 'se', 'S00'),
            ("What did the person do from {} seconds to {} seconds? List the things they do sequentially.", 'se', 'S01'),
            ("The person did A {} B between {} seconds and {} seconds. What are A and B?", 'bse', 'S02'),
            ("Focus on the segment {} seconds - {} seconds. What did the person do {} they {}?", 'sebj', 'S03'),
            ("Answer the above question according to the video. Only use words from the following words to organize your answer.", '', 'S04',),
        ],
        [
            ("Which object did the person {} {} they {} ___? ___.", 'vbd', 'S10'),
            ("The person ___ {} they ___.", 'b', 'S11'),
            ("Choose words from the following words to fill in the blanks according to segment {} seconds - {} seconds of the video.", 'se', 'S12'),
        ]
    ],
    "Prediction":[
        ("What will the person do after {} seconds - {} seconds?", "se", 'P0',),
        ("What will the person do next with the {} after {} seconds - {} seconds?", "nse", 'P1',),
        ("Which object would the person {} after {} seconds - {} seconds?", "vse", 'P2'),
        ("According to {} seconds - {} seconds, which object would the person {} next after they {} the {}?", "", 'P3',),
        ("Choose answer from the following options.", "", 'P4',),
        ("Only use words from the following words to organize your answer.", "", 'P5',),
    ]
}