for b in '10000 30000 50000';
do
    for r in '0.005 0.01 0.02';
    do
        python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $b -lr $r --exp_name hc_b$b_r$r
    done
done
