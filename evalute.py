from jsonargparse import auto_cli
import torch
from transformers import AutoTokenizer
import mteb
from mteb.cache import ResultCache
from models import MiniLMSentenceTransformer


def main(
        model_revision: str = None,
        load_path: str = None
):
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    model = MiniLMSentenceTransformer(vocab_size=tokenizer.vocab_size)
    model.set_revision(model_revision)

    state_dict = torch.load(
        f"{load_path}/{model_revision}.pt", 
        map_location="cpu", 
        weights_only=True
    )
    model.load_state_dict(state_dict)

    sts_tasks = ['HUMESICK-R', 'SynPerSTS', 'JSICK', 'JSTS', 'CDSC-R']

    cache = ResultCache(cache_path="./")
    sts_results = mteb.evaluate(
        model,
        tasks=mteb.get_tasks(tasks=sts_tasks),
        cache=cache
    )
    sts_scores = []
    for sts_result in sts_results.task_results:
        main_socres = []
        score = sts_result.scores
        if "validation" in score:
            for i in score["validation"]:
                main_socres.append(i["main_score"])
        if "train" in score:
            for i in score["train"]:
                main_socres.append(i["main_score"])
        if "test" in score:
            for i in score["test"]:
                main_socres.append(i["main_score"])
        main_score = sum(main_socres)/len(main_socres)
        sts_scores.append(main_score)
        print(f"Task: {sts_result.task_name:<15} | Score: {main_score:>8.4f}")
    print(f"======  Average Score >>> {sum(sts_scores)/len(sts_scores):.4f}  ======")


if __name__ == '__main__':
    auto_cli(main)