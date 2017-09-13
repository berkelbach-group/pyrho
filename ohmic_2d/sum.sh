#!/bin/bash
for L in 1 3 4; do
    for t2 in 10.0 625.0; do
        paste 2d_allesa_HEOM_t2-${t2}_dt-10_tf-1000_L-${L}_K-0.dat 2d_allgsb_HEOM_t2-${t2}_dt-10_tf-1000_L-${L}_K-0.dat 2d_allese_HEOM_t2-${t2}_dt-10_tf-1000_L-${L}_K-0.dat | awk '{ printf("%0.8f %0.8f %0.8f \n", $1, $2, $3+$6+$9); }' > 2d_total_HEOM_t2-${t2}_dt-10_tf-1000_L-${L}_K-0.dat;

         sed 's/0.00000000 0.00000000 0.00000000/ /g' 2d_total_HEOM_t2-${t2}_dt-10_tf-1000_L-${L}_K-0.dat > temp2.dat && mv temp2.dat 2d_total_HEOM_t2-${t2}_dt-10_tf-1000_L-${L}_K-0.dat
    
    
    
    done
done
