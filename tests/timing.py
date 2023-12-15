import time
import replicate
import os

base = "replicate-internal/staging-llama-2-7b:8ba7b9478e1cbdde020f79f0838cd94465dfc6fc0207e01d2e59c00422f65148"

v1 = "a42037aa39fc7cdc9138d61a0a94172107906ed8be7c8b0568cc5766d633f0fe"
v2 = "ca0a7d930eed4f330d7f187a18052842f35087fc15b93b741a554753591cb366"

model = replicate.models.get("technillogue/llama2-summarizer")
ver1 = model.versions.get(v1)
ver2 = model.versions.get(v2)

os.system("kubectl delete pod -l replicate/version_short_id=8ba7b947")


def run(v):
    t0 = time.time()
    # gen = replicate.run(v1, input={"prompt": "a"})
    global last
    last = pred = replicate.predictions.create(v, input={"prompt": "a"})
    t1 = time.time()
    print(f"got result after {t1 - t0:.4f}")
    gen = pred.output_iterator()
    next(gen)
    t2 = time.time()
    print(f"got first token {t2 - t1:.4f}")
    try:
        print(re.search("previous weights were (.*)\n", pred.logs).group().strip())
    except:
        pass
    try:
        print(re.search("Downloaded peft weights in (\d+.\d+)", pred.logs).group())
    except:
        pass
    try:
        print(re.search("initialize_peft took (\d+.\d+)", pred.logs).group())
    except:
        pass
    print(f"prediciton created to first token: {t2 - t0:.4f}")
    pred.wait()
    t3 = time.time()
    print(re.search("hostname: (.*)\n", pred.logs).group().strip())
    print(f"prediction took {t3 - t2:.4f} from first to last token")


run(ver1)
run(ver2)
