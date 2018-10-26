python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $1 -lr $2 -rtg --exp_name hc_b$1_r$2
