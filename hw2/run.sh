# python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $1 -lr $2 --exp_name hc_b$1_r$2
# python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $1 -lr $2 -rtg --exp_name hc_b$1_r$2
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $1 -lr $2 --nn_baseline --exp_name hc_b$1_r$2-bl
# python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $1 -lr $2 -rtg --nn_baseline --exp_name hc_b$1_r$2_rtg_bl
