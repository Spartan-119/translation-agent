import sacrebleu
from bert_score import score
from comet import download_model, load_from_checkpoint

# Define source texts, human translation, and two machine translations
source_texts = [
    "Want some free spins? Simply bet on any slot to receive a mystery amount every day http://nbet.co/mba-en T&Cs apply. Opt out: eu.stp3.co What Will the Mystery Box Bring You? Win up to 75 Free Spins daily with the Mystery Box Available on Fire Joker T&cs Apply https://casino.netbet.com/en/mystery-box-august24",
    "Will you win 75 Free Spins [%FIRST_NAME%]?"
]

human_translations = [
    "Lust auf ein paar Freispiele? Setze einfach auf einen beliebigen Slot und erhalte jeden Tag einen geheimnisvollen Betrag http://nbet.co/mba-de AGB gelten. Abmelden: http://de.stp3.co/ Was wird die Mystery Box dir schenken? Gewinne jeden Tag bis zu 75 Freispiele mit der Mystery Box Nur auf Fire Joker AGB gelten https://www.netbet.de/mystery-box-ago24",
    "Gewinnst du 75 Freispiele, [%FIRST_NAME%]?"
]

# Two different machine translations
ai_agent = [
    "Möchtest du 75 Freispiele gewinnen? Setze einfach bei einem beliebigen Slot, um täglich einen geheimen Betrag zu erhalten http://nbet.co/mba-en AGB gelten. Vom Newsletter abmelden: eu.stp3.co Was wird die Mystery Box (Geheimnisbox) Ihnen bringen? Gewinne täglich bis zu 75 Freispiele mit der Mystery Box Verfügbar bei Fire Joker, AGB gelten https://casino.netbet.com/en/mystery-box-august24",
    "Wirst du 75 Freispiele gewinnen, [%FIRST_NAME%]?"
]

deepl = [
    "Möchten Sie ein paar Freispiele? Setzen Sie einfach auf einen beliebigen Spielautomaten und Sie erhalten jeden Tag einen geheimnisvollen Betrag. http://nbet.co/mba-en Es gelten die allgemeinen Geschäftsbedingungen. Abmeldung: eu.stp3.co Was bringt Ihnen die Mystery Box? Gewinnen Sie täglich bis zu 75 Freispiele mit der Mystery Box Erhältlich bei Fire Joker. Es gelten die AGBs https://casino.netbet.com/en/mystery-box-august24",
    "Gewinnen Sie 75 Free Spins [%FIRST_NAME%]?"
]

# Function to evaluate a model's translation
def evaluate_translation(machine_translations, model_name):
    print(f"\nEvaluating {model_name}...")

    # BLEU Score
    bleu = sacrebleu.corpus_bleu(machine_translations, [human_translations])
    print(f"BLEU Score: {bleu.score}")

    # TER Score
    ter = sacrebleu.corpus_ter(machine_translations, [human_translations])
    print(f"TER Score: {ter.score}")

    # # METEOR Score
    # meteor = sacrebleu.meteor_score(machine_translations, [human_translations])
    # print(f"METEOR Score: {meteor}")

    # # BERTScore
    # P, R, F1 = score(machine_translations, human_translations, lang="de")
    # print(f"BERTScore F1: {F1.mean().item()}")

    # # COMET Score
    # model_path = download_model("Unbabel/wmt22-comet-da")
    # comet_model = load_from_checkpoint(model_path)
    # data = [{"src": s, "mt": mt, "ref": ht} for s, mt, ht in zip(source_texts, machine_translations, human_translations)]
    # comet_score = comet_model.predict(data, batch_size=1)
    # print(f"COMET Score: {sum(comet_score['scores']) / len(comet_score['scores'])}")

# Evaluate both machine translations
evaluate_translation(ai_agent, "Machine Translation AI Agent")
evaluate_translation(deepl, "Machine Translation DeepL")
