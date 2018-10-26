import os

for b in (10000, 30000, 50000):
    for r in (0.005, 0.01, 0.02):
        os.system("python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {b} -lr {r} --exp_name hc_b{b}_r{r}".format(b=b, r=r))
