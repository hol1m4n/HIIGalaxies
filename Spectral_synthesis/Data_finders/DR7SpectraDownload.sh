#!/bin/bash

cd $HOME/Data_storageHII/DR7SpectraDbox/


cd p1/p1/ 
pwd
mv *.fit ../../
cd ../../

cd p2/p2/ 
pwd
mv *.fit ../../
cd ../../

cd p3/p3/ 
pwd
mv *.fit ../../
cd ../../

rmdir p1/p1/
rmdir p2/p2/
rmdir p3/p3/

rmdir p1
rmdir p2
rmdir p3

pwd

ls | wc -l





echo "Proceso finalizado."




