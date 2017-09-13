#!/bin/bash
for i in 0.0 100.0 200.0 300.0 400.0 500.0 600.0; do
    for j in 1.0 2.0 3.0 4.0 5.0 6.0 8.0 10.0 12.0; do
        FILENAME=2d_hybrid_t2-${i}_dt-4_tf-500_tauc100.0_lamda50.0_T300.0_split${j}_average1e4.dat
        awk '{print;} NR % 161 == 0 { print ""; }' $FILENAME > temp.dat && mv temp.dat $FILENAME && sed '/0.000000000 0.000000000 0.000000000/d' $FILENAME > temp2.dat && mv temp2.dat $FILENAME ;
    done
done
