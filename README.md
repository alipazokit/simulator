# Simulator
Efficent Simulator


## Prerequisites
The following packages are required on a linux machine to compile and use the software package.
```
g++
cmake
make
```


```
git clone https://github.com/alipazokit/simulator.git
cd simulator
mkdir build
cd build/
cmake ..
make

```
There is a toy example in the example folder (please take a look at test.sh file ).

## Input arguments 
```
-g :
 genotype file
-annot :
 annotation file (in case of single variance component, just provide a single column where all elements are equal to 1)
-o :
 output directory 
-maf_ld :
 the file which includes MAF and LD of SNPs. 
-simul_par :
 the  file which contains simulation parameters used in ((Eq.12)  in our paper ) . It has seven columns (percentage of causal SNPs, exponent of LD,exponent of MAF, min(MAF) of causal SNPs, max(MAF) of causal SNPs, total h2, number of simulations).
 ```
