"""Chat topic classifier for the retention draw (docs/DATA_PLAN.md section 3.4): nearest prototype by cosine.

Each topic has two or three short prototype descriptions. A prompt and the prototypes are embedded with
BAAI/bge-small-en-v1.5 through the D3/D5' embedder (`screens.d3_match.Embedder`: mean pooling, L2-normalised, the
local snapshot at `paths.BGE_SMALL_DIR`); the prompt's topic is the topic of its most similar prototype. Topics belong
to two groups: `hss` (humanities, social science, law, economics, history, linguistics, arts and general knowledge,
the group the plan requires to be at least 40 percent of the chat rows) and `other`.

The classifier is checked on a 40-prompt sample labelled by hand (`topic_check_labels.json`, task ids and labels only);
the builder reports its group confusion and per-topic agreement in the pool manifest.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

CLASSIFIER_VERSION = "topics-proto-v1"
HSS, OTHER = "hss", "other"
PROTOTYPES: dict[str, tuple[str, list[str]]] = {
    "history": (HSS, ["A question about history: historical events, wars, empires, dynasties and revolutions.",
                      "Who was this historical figure and what happened in that period of the past?",
                      "Ancient civilizations, medieval times, the world wars and the history of a country."]),
    "law": (HSS, ["A legal question about laws, courts, contracts, legal rights and regulations.",
                  "Is this legal? What does the law say about a lawsuit, a crime, a lease or an employment dispute?",
                  "Constitutional law, criminal justice, intellectual property and legal procedure."]),
    "economics_business": (HSS, ["A question about economics, markets, inflation, trade and monetary policy.",
                                 "Business strategy, management, marketing, accounting, finance and investing.",
                                 "How do interest rates, taxes, stocks and companies work?"]),
    "social_science": (HSS, ["A question about psychology, sociology, anthropology or human behaviour.",
                             "Politics, government, elections, political ideologies and international relations.",
                             "Education, society, culture wars, social issues and public policy debates."]),
    "linguistics_language": (HSS, ["A question about language and linguistics: grammar, syntax and word meanings.",
                                   "Etymology, the origin of a word or phrase, idioms, dialects and pronunciation.",
                                   "How do you say this in another language, and what is the difference between two words?"]),
    "philosophy_religion": (HSS, ["A question about philosophy, ethics and moral dilemmas.",
                                  "Religion, theology, mythology, scripture and spiritual beliefs.",
                                  "What did this philosopher argue about free will, knowledge or the meaning of life?"]),
    "arts_literature": (HSS, ["A question about literature, novels, authors, poetry and literary analysis.",
                              "Music, film, television, painting, art history, architecture and theatre.",
                              "Analyse the themes and characters of a book, a movie or a song."]),
    "geography_knowledge": (HSS, ["A general knowledge or trivia question about countries, cities and geography.",
                                  "Facts about world cultures, traditions, famous places, sports history and records.",
                                  "Which country, what is the capital, how many, who holds the record?"]),
    "coding": (OTHER, ["A programming task: write, debug or explain code.",
                       "Python, JavaScript, SQL, an API, a software bug, an error message or a stack trace.",
                       "Software engineering, algorithms, data structures, scripts and command-line tools."]),
    "math": (OTHER, ["A math problem: arithmetic, algebra, calculus, probability or geometry.",
                     "Solve this equation, compute the value, prove the statement.",
                     "A word problem with numbers, percentages, rates and units."]),
    "science_tech": (OTHER, ["A question about natural science: physics, chemistry, biology and astronomy.",
                             "Medicine, health, symptoms, diseases, nutrition and the human body.",
                             "Engineering, electronics, computers, gadgets, AI and technology products."]),
    "creative_writing": (OTHER, ["A creative writing request: write a story, a poem, song lyrics or a screenplay.",
                                 "Roleplay as a character, continue the fiction, write a funny dialogue or a joke.",
                                 "Generate names, slogans, a fantasy world or a plot for a game."]),
    "practical_writing": (OTHER, ["Write or edit an email, a cover letter, a resume, a report or marketing copy.",
                                  "Rewrite, paraphrase, summarise, proofread or translate this text.",
                                  "Draft a social media post, a product description or a speech."]),
    "advice_lifestyle": (OTHER, ["Personal advice about relationships, careers, motivation and mental wellbeing.",
                                 "Recommendations for cooking, recipes, travel plans, shopping, fitness and hobbies.",
                                 "How should I handle this everyday situation?"]),
    "puzzles_games": (OTHER, ["A riddle, a logic puzzle, a trick question or a brain teaser.",
                              "Play a word game, a guessing game or a text adventure with me.",
                              "Count the letters in a word, or answer a tricky reasoning question."]),
    "assistant_chitchat": (OTHER, ["Casual conversation, a greeting, small talk.",
                                   "Questions about the AI assistant itself: who are you, what can you do?",
                                   "Tell me something interesting, or test the chatbot."]),
}
TOPICS = tuple(PROTOTYPES)
GROUP = {t: g for t, (g, _) in PROTOTYPES.items()}
LABELS_PATH = Path(__file__).with_name("topic_check_labels.json")


class TopicClassifier:
    """Nearest-prototype topic classifier over an embedder with `embed(texts) -> L2-normalised ndarray`."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        texts, owners = [], []
        for topic, (_, descs) in PROTOTYPES.items():
            texts += descs
            owners += [topic] * len(descs)
        self.proto = np.asarray(embedder.embed(texts), dtype=np.float32)
        self.owners = owners

    def classify(self, prompts: Sequence[str]) -> list[dict[str, Any]]:
        """Per prompt: topic, group, cosine of the best prototype and the margin to the best prototype of the other
        group."""
        if not prompts:
            return []
        sims = np.asarray(self.embedder.embed(list(prompts)), dtype=np.float32) @ self.proto.T
        groups = np.array([GROUP[o] for o in self.owners])
        out = []
        for row in sims:
            j = int(row.argmax())
            topic = self.owners[j]
            other_best = float(row[groups != GROUP[topic]].max())
            out.append({"topic": topic, "group": GROUP[topic], "cosine": round(float(row[j]), 4),
                        "group_margin": round(float(row[j]) - other_best, 4)})
        return out


def load_labels(path: Path = LABELS_PATH) -> dict[str, dict[str, str]]:
    """task_id -> {"topic", "group"} of the hand-labelled check sample."""
    d = json.loads(path.read_text())
    return {k: {"topic": v, "group": GROUP[v]} for k, v in d["labels"].items()}


def check_report(predicted: dict[str, dict[str, Any]], labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Group confusion (hand label x prediction), group accuracy, topic agreement, and the hss precision and recall."""
    ids = [k for k in labels if k in predicted]
    conf: Counter = Counter((labels[k]["group"], predicted[k]["group"]) for k in ids)
    n = len(ids)
    tp, fp, fn = conf[(HSS, HSS)], conf[(OTHER, HSS)], conf[(HSS, OTHER)]
    return {"n_labelled": len(labels), "n_scored": n, "missing": sorted(set(labels) - set(ids)),
            "group_confusion": {f"label_{a}__pred_{b}": conf[(a, b)] for a in (HSS, OTHER) for b in (HSS, OTHER)},
            "group_accuracy": round((conf[(HSS, HSS)] + conf[(OTHER, OTHER)]) / n, 4) if n else None,
            "hss_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "hss_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "topic_agreement": round(sum(labels[k]["topic"] == predicted[k]["topic"] for k in ids) / n, 4) if n else None,
            "label_hss_share": round(sum(labels[k]["group"] == HSS for k in ids) / n, 4) if n else None,
            "pred_hss_share": round(sum(predicted[k]["group"] == HSS for k in ids) / n, 4) if n else None}
