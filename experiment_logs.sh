#! /bin/bash


####################################################
# Section 4.1 Supervised Coordinate Classification
####################################################
## All experiments for Supervised Coordinate Classification, Deconv, uniform and quarant split
for lr in {0.001,0.002,0.005,0.01,0.02,0.05}
do
    for l2reg in {0.,0.001,0.01}
    do
        for fs in {2,3,4}
        do
            for mul in {1,2,3}
            do
                for minibatch in{16,32}
                do
                    EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
                    echo "$EXP_ID resman -d results/deconv_uniform_SCC/ -r model${mul}x_mb${minibatch}_LR${lr}_l2reg${l2reg}_fs${fs} -- ./train.py --arch deconv_classification -mb ${minibatch} -E 1000 -L $lr --opt adam --l2 $l2reg -mul $mul -fs ${fs} --lrpolicy step --lrstepratio 0.1 --lrstepevery 200 --lrmaxsteps 4 --data_h5 data/rectangle_4_uniform.h5" >> experiments_SCC_deconv_uniform.txt
                    EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
                    echo "$EXP_ID resman -d results/deconv_quadrant_SCC/ -r model${mul}x_mb${minibatch}_LR${lr}_l2reg${l2reg}_fs${fs} -- ./train.py --arch deconv_classification -mb ${minibatch} -E 1000 -L $lr --opt adam --l2 $l2reg -mul $mul -fs ${fs} --lrpolicy step --lrstepratio 0.1 --lrstepevery 200 --lrmaxsteps 4 --data_h5 data/rectangle_4_quadrant.h5" >> experiments_SCC_deconv_quadrant.txt
                done
            done
       done
    done
done

## All experiments for Supervised Coordinate Classification, CoordConv, uniform and quarant split
for lr in {0.001,0.005,0.01}
do
    for l2reg in {0.,0.001}
    do
        EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
        echo "$EXP_ID resman -d results/coordconv_SCC/ -r model1x_mb16_LR${lr}_l2reg${l2reg}_uniform -- ./train.py --arch coordconv_classification -mb 16 -E 50 -L $lr --opt adam --l2 $l2reg -mul 1 -fs 1 --data_h5 data/rectangle_4_uniform.h5" >> experiments_SCC_coordconv_uniform.txt
        EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
        echo "$EXP_ID resman -d results/coordconv_SCC/ -r model1x_mb16_LR${lr}_l2reg${l2reg}_quadrant -- ./train.py --arch coordconv_classification -mb 16 -E 50 -L $lr --opt adam --l2 $l2reg -mul 1 -fs 1 --data_h5 data/rectangle_4_quarant.h5" >> experiments_SCC_coordconv_quadrant.txt
    done
done



####################################################
# Section 4.2 Supervised Coordinate Regression
####################################################
## All experiments for Supervised Coordinate Regression, Conv, both splits
for lr in {0.0005,0.001,0.01}
do
    for l2reg in {1e-5,1e-4,5e-4}
    do
        EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
        echo "$EXP_ID resman -d results/conv_SCR/ -r model_LR${lr}_l2reg${l2reg}_uniform -- ./train.py --arch conv_regressor -E 100 -L $lr --opt adam --l2 $l2reg -mul 1 --data_h5 data/rectangle_4_uniform.h5" >> experiments_SCR_conv_uniform.txt
        EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
        echo "$EXP_ID resman -d results/conv_SCR/ -r model_LR${lr}_l2reg${l2reg}_uniform -- ./train.py --arch conv_regressor -E 100 -L $lr --opt adam --l2 $l2reg -mul 1 --data_h5 data/rectangle_4_quarant.h5" >> experiments_SCR_conv_quadrant.txt
    done
done


## CoordConv, both splits
CUDA_VISIBLE_DEVICES=0 ./train.py --epochs 100 --arch coordconv_regressor --data_h5 data/rectangle_4_uniform.h5 --lr 0.01 --opt adam --l2 0.00001
CUDA_VISIBLE_DEVICES=0 ./train.py --epochs 100 --arch coordconv_regressor --data_h5 data/rectangle_4_quarant.h5 --lr 0.01 --opt adam --l2 0.00001


####################################################
# Section 4.3 Supervised Rendering
####################################################
## All experiments for Supervised Rendering, Deconv, uniform and quarant split
for lr in {0.001,0.002,0.005,0.01,0.02,0.05}
do
    for l2reg in {0.,0.001,0.01}
    do
        for fs in {2,3,4}
        do
            for mul in {1,2,3}
            do
                for minibatch in {16,32}
                do
                    for loss in {sig,mse}
                    do
                        EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
                        echo "$EXP_ID resman -d results/deconv_uniform_SR/ -r model${mul}x_mb${minibatch}_LR${lr}_l2reg${l2reg}_fs${fs} -- ./train.py --arch deconv_rendering -mb ${minibatch} -E 1000 -L $lr --opt adam --l2 $l2reg -mul $mul -fs ${fs} --lrpolicy step --lrstepratio 0.1 --lrstepevery 200 --lrmaxsteps 4 -${loss} --data_h5 data/rectangle_4_uniform.h5" >> experiments_SR_deconv_uniform.txt
                        EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
                        echo "$EXP_ID resman -d results/deconv_quadrant_SR/ -r model${mul}x_mb${minibatch}_LR${lr}_l2reg${l2reg}_fs${fs} -- ./train.py --arch deconv_rendering -mb ${minibatch} -E 1000 -L $lr --opt adam --l2 $l2reg -mul $mul -fs ${fs} --lrpolicy step --lrstepratio 0.1 --lrstepevery 200 --lrmaxsteps 4 -${loss} --data_h5 data/rectangle_4_quadrant.h5" >> experiments_SR_deconv_quadrant.txt
                    done
                done
            done
       done
    done
done

## All experiments for Supervised Coordinate Classification, CoordConv, uniform and quarant split
for lr in {0.001,0.005}
do
    for l2reg in {0.,0.001}
    do
        for mul in{1,2}
        do
            EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
            echo "$EXP_ID resman -d results/coordconv_SR/ -r model1x_mb16_LR${lr}_l2reg${l2reg}_uniform -- ./train.py --arch coordconv_rendering -mb 16 -E 50 -L $lr --opt adam --l2 $l2reg -mul $mul -fs 1 --data_h5 data/rectangle_4_uniform.h5" >> experiments_SR_coordconv_uniform.txt
            EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
            echo "$EXP_ID resman -d results/coordconv_SR/ -r model1x_mb16_LR${lr}_l2reg${l2reg}_quadrant -- ./train.py --arch coordconv_rendering -mb 16 -E 50 -L $lr --opt adam --l2 $l2reg -mul $mul -fs 1 --data_h5 data/rectangle_4_quarant.h5" >> experiments_SR_coordconv_quadrant.txt
        done
    done
done


