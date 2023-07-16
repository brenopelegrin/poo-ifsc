#!/bin/bash
python3 main.py input/chicken.pgm 5 > test_chicken_01.out && diff -s output/test_chicken_01.out test_chicken_01.out &&
python3 main.py input/chicken.pgm 128 > test_chicken_02.out && diff -s output/test_chicken_02.out test_chicken_02.out &&
python3 main.py input/chicken.pgm 51 > test_chicken_03.out && diff -s output/test_chicken_03.out test_chicken_03.out &&
python3 main.py myinputs/chicken.pgm 51 > test_custom_chicken_03.out && diff -s output/test_custom_chicken_03.out test_custom_chicken_03.out &&
python3 main.py input/mule.pgm 1 > test_mule_01.out && diff -s output/test_mule_01.out test_mule_01.out &&
python3 main.py input/mule.pgm 64 > test_mule_02.out && diff -s output/test_mule_02.out test_mule_02.out &&
python3 main.py input/mule.pgm 256 > test_mule_03.out && diff -s output/test_mule_03.out test_mule_03.out &&
python3 main.py input/owl.pgm 32 > test_owl_01.out && diff -s output/test_owl_01.out test_owl_01.out &&
python3 main.py input/owl.pgm 300 > test_owl_02.out && diff -s output/test_owl_02.out test_owl_02.out &&
python3 main.py input/owl.pgm 0 > test_owl_03.out && diff -s output/test_owl_03.out test_owl_03.out